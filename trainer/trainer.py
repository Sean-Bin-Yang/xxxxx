import os
import torch
import logging
import numpy as np
import torch.nn as nn

from tqdm import tqdm
import time
import datetime

SMALL_NUM = np.log(1e-45)


class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.5, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)

class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()
       
        self.log_vars = nn.Parameter(torch.zeros(2))  

    def forward(self, loss1, loss2):
        loss = (
            0.5 * torch.exp(-self.log_vars[0]) * loss1 + self.log_vars[0] +
            0.5 * torch.exp(-self.log_vars[1]) * loss2 + self.log_vars[1]
        )
        return loss

def compute_mgda_alphas(grads):
    grads = torch.stack(grads)  # [num_losses, num_params]
    GG = torch.matmul(grads, grads.t())  # Gram matrix
    k = grads.size(0)
    device = grads.device

    try:
        if k == 2:
            g1, g2 = GG[0, 0], GG[1, 1]
            g12 = GG[0, 1]
            if g12 >= g1:
                return torch.tensor([1.0, 0.0], device=device)
            if g12 >= g2:
                return torch.tensor([0.0, 1.0], device=device)
            alpha1 = (g2 - g12) / (g1 + g2 - 2 * g12)
            return torch.tensor([alpha1, 1 - alpha1], device=device)

        else:
            import cvxpy as cp
            import numpy as np

            GG_np = GG.detach().cpu().numpy()

            # 
            GG_np += 1e-4 * np.eye(k)

            alpha = cp.Variable(k)
            objective = cp.Minimize(0.5 * cp.quad_form(alpha, GG_np))
            constraints = [cp.sum(alpha) == 1, alpha >= 0]
            prob = cp.Problem(objective, constraints)

            # 
            prob.solve(solver=cp.SCS, verbose=False)

            if alpha.value is None or np.any(np.isnan(alpha.value)):
                raise ValueError("CVXPY failed to solve MGDA QP.")

            return torch.tensor(alpha.value, dtype=torch.float32, device=device)

    except Exception as e:
        print(f"[MGDA] QP solver failed: {e}")
        print("[MGDA] Falling back to uniform weights.")
        return torch.ones(k, device=device) / k

class REFINETrainer:
    def __init__(self,
                 config,
                 model,
                 node_feature,
                 edge_index,
                 train_dataloader
                 ):
        super().__init__()
        self.clip = config['clip']
        self.lr = config['lr']
        self.betas = eval(config['betas'])
        self.weight_decay = config['weight_decay']
        self.device = config['device']
        self.epochs = config['epochs']
        self.dim = config['enc_embed_dim']
        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']


        ###adaptive temperature
        self.adj_tau = config['adj_tau']
        self.t_mat = config['t_max']
        self.max_tau =config['temperature_max']
        self.min_tau = config['temperature_min']
        self.temperature=config['temperature']
        self.enc_depths=config['enc_depths']
    
        
        self.node_feature = node_feature
        self.edge_index = edge_index
        self.train_dataloader = train_dataloader


        # self.loss_dcl=DCL()
        # self.dcl=DCL(temperature=0.5)
        self.loss_dclw = DCLW(sigma=0.5, temperature=0.5)
        self.multi_loss_optimize = MultiTaskLossWrapper()

        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay,
                                           betas=self.betas)

        self.loss_fn = nn.NLLLoss(ignore_index=0)
        self.save_path = os.path.join('checkpoints', config['dataset'],config['method'], config['exp_id'], 'pretraining-refine')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def iteration(self, epoch, dataloader, iteration_type):
        log_losses =[]

        pbar = tqdm(dataloader)
        for batch_data in pbar:
            enc_data, dec_data, y1, y2 = batch_data
            enc_data = [data.to(self.device) for data in enc_data]
            dec_data = [data.to(self.device) for data in dec_data]
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)
            node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
            edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

            self.optimizer.zero_grad()
            
            pred1, pred2, z, z_hat= self.model(node_feature, edge_index, enc_data, dec_data, y2, self.lambda2)
            
            loss_pre=self.loss_fn(pred2.transpose(1, 2), y2)

            local_dcl =[]
            losses =[]
            g_z =z.mean(dim=1).squeeze(1)
            g_z_hat=z_hat.mean(dim=1).squeeze(1)
            loss_dcl = self.loss_dclw(g_z, g_z_hat)
            tloss = self.multi_loss_optimize(loss_pre,loss_dcl)
            tloss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            loss_val = torch.nn.functional.softplus(tloss.detach()).cpu().item()
            pbar.set_description('[{} Epoch {}/{}: loss: %f]'.format(iteration_type, str(epoch), str(self.epochs)) % tloss)
            losses.append(tloss.item())
        return np.array(losses).mean()

    def train(self):
        start_time = time.time()
        # enc_depths=config['enc_depths']
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = self.iteration(epoch, self.train_dataloader, 'train')
            logging.info(f'Epoch {epoch}/{self.epochs}, avg train loss: {train_loss}')
            torch.save(self.model.state_dict(), f'{self.save_path}/pretraining_{self.enc_depths}_{epoch + 1}.pt')
            if epoch % 2 ==0 or epoch +1 ==self.epochs:
                total_time1 = time.time() - start_time
                total_time_str1 =str(epoch)+'_'+str(datetime.timedelta(seconds=int(total_time1)))
                with open('./time_save_result-dep.txt', 'a', encoding = 'utf-8') as f:
                    f.write(total_time_str1)
                    f.write('\n')
