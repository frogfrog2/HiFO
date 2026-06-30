import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0, return_mean=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.return_mean = return_mean

    def forward(self, a, b=0):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        if self.return_mean:
            return self.loss_weight * torch.mean(loss)
        else:
            return self.loss_weight * torch.sum(loss)

@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = F.mse_loss(a, b)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class SURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity
    """

    def __init__(self, w_bij=1.0, w_orth=1.0, w_lap=1e-3):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
        """
        super(SURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap

    def forward(self, C12, C21, evals_1, evals_2):
        """
        Compute bijectivity loss + orthogonality loss
                            + Laplacian commutativity loss
                            + descriptor preservation via commutativity loss

        Args:
            C12 (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            C21 (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            evals_1 (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_2 (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        criterion = SquaredFrobeniusLoss()
        eye = torch.eye(C12.shape[1], C12.shape[2], device=C12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C12.shape[0], dim=0)

        losses = dict()
        # Bijectivity penalty
        if self.w_bij > 0:
            bijectivity_loss = criterion(torch.bmm(C12, C21), eye_batch) + criterion(torch.bmm(C21, C12), eye_batch)
            bijectivity_loss *= self.w_bij
            losses['l_bij'] = bijectivity_loss

        # Orthogonality penalty
        if self.w_orth > 0:
            orthogonality_loss = criterion(torch.bmm(C12.transpose(1, 2), C12), eye_batch) + \
                                 criterion(torch.bmm(C21.transpose(1, 2), C21), eye_batch)
            orthogonality_loss *= self.w_orth
            losses['l_orth'] = orthogonality_loss

        # Laplacian commutativity penalty
        if self.w_lap > 0:
            laplacian_loss = criterion(torch.einsum('abc,ac->abc', C12, evals_1),
                                       torch.einsum('ab,abc->abc', evals_2, C12))
            laplacian_loss += criterion(torch.einsum('abc,ac->abc', C21, evals_2),
                                        torch.einsum('ab,abc->abc', evals_1, C21))
           # laplacian_loss *= self.w_lap
            losses['l_lap'] = laplacian_loss

        return losses

@LOSS_REGISTRY.register()
class ClusterLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ClusterLoss, self).__init__()
        self.loss_weight = loss_weight

    #def forward(self, pxy, pyx, faces_x, faces_y, epoch):
        #mx_row = torch.argmax(pxy, dim=1) # 2014
        #my_row = torch.argmax(pyx, dim=1) # 2060
        #mx_row_uniq = torch.unique(mx_row) # 0~2059
        #my_row_uniq = torch.unique(my_row) # 0~2013
        #matchx = torch.ones_like(my_row) # 2060
        #matchy = torch.ones_like(mx_row) # 2014
        #matchx[mx_row_uniq] = 0.0
        #matchy[my_row_uniq] = 0.0
        #edges_x = torch.unique(torch.cat([faces_x[:, [0,1]], faces_x[:, [1,0]],faces_x[:, [1,2]], faces_x[:, [2,1]], faces_x[:, [2,0]], faces_x[:, [0,2]]], dim=0), dim=0)
        #edges_y = torch.unique(torch.cat([faces_y[:, [0,1]], faces_y[:, [1,0]],faces_y[:, [1,2]], faces_y[:, [2,1]], faces_y[:, [2,0]], faces_y[:, [0,2]]], dim=0), dim=0)
        #srcx = edges_x[:, 0]
        #dstx = edges_x[:, 1]
        #srcy = edges_y[:, 0]
        #dsty = edges_y[:, 1]
        #defx_src = matchx[srcy]
        #defx_dst = matchx[dsty]
        #defy_src = matchy[srcx]
        #defy_dst = matchy[dstx]
        #lossx = torch.sum(defx_src * defx_dst)
        #lossy = torch.sum(defy_src * defy_dst)
        #return self.loss_weight * (lossx + lossy)
        #mx_row, _ = torch.max(pxy, dim=1, keepdim=True)
        #my_row, _ = torch.max(pyx, dim=1, keepdim=True)
        #maxrx = pxy / mx_row
        #maxry = pyx / my_row
        #probx = 1 - maxrx
        #proby = 1 - maxry
        #defix = probx.prod(dim=0) # (n_y)
        #defiy = proby.prod(dim=0)
        #edges_x = torch.unique(torch.cat([faces_x[:, [0,1]], faces_x[:, [1,0]],faces_x[:, [1,2]], faces_x[:, [2,1]], faces_x[:, [2,0]], faces_x[:, [0,2]]], dim=0), dim=0)
        #edges_y = torch.unique(torch.cat([faces_y[:, [0,1]], faces_y[:, [1,0]],faces_y[:, [1,2]], faces_y[:, [2,1]], faces_y[:, [2,0]], faces_y[:, [0,2]]], dim=0), dim=0)
        #srcx = edges_x[:, 0]
        #dstx = edges_x[:, 1]
        #srcy = edges_y[:, 0]
        #dsty = edges_y[:, 1]
        #defx_src = defix[srcy]
        #defx_dst = defix[dsty]
        #defy_src = defiy[srcx]
        #defy_dst = defiy[dstx]
        #lossx = torch.sum(defix) #lossx = torch.sum(defx_src * defx_dst)
        #lossy = torch.sum(defiy) #lossy = torch.sum(defy_src * defy_dst)
        #return self.loss_weight * (lossx + lossy)
#    def forward(self, pxy, pyx, faces_x, faces_y, threshold=0.04, scale=50.0):
#        #demandy = torch.sum(pxy, dim=0)
#        demandy, _ = torch.max(pxy, dim=0) # (n_y)
#        #demandx = torch.sum(pyx, dim=0)
#        demandx, _ = torch.max(pyx, dim=0) # (n_x)
#        #print(demandx, demandy)
#        zy = (threshold - demandy) * scale
#        zx = (threshold - demandx) * scale
#        my = F.softplus(zy, beta=20.0) # (n_y)
#        mx = F.softplus(zx, beta=20.0) # (n_x)
#        edges_x = torch.unique(torch.cat([faces_x[:, [0,1]], faces_x[:, [1,2]], faces_x[:, [2,0]]], dim=0), dim=0)
#        edges_y = torch.unique(torch.cat([faces_y[:, [0,1]], faces_y[:, [1,2]], faces_y[:, [2,0]]], dim=0), dim=0)
#        srcx = edges_x[:, 0]
#        dstx = edges_x[:, 1]
#        srcy = edges_y[:, 0]
#        dsty = edges_y[:, 1]
#        mx_src = mx[srcx]
#        mx_dst = mx[dstx]
#        my_src = my[srcy]
#        my_dst = my[dsty]
#        lossx = torch.sum(mx) #lossx = torch.sum(mx_src * mx_dst)
#        lossy = torch.sum(my) #lossy = torch.sum(my_src * my_dst)
    def forward(self, pxy, pyx, faces_x, faces_y, epoch, threshold=0.0001):
        #col_sumx = torch.sum(pxy*pxy, dim=0) # ny
        #col_sumy = torch.sum(pyx*pyx, dim=0)
        #col_mx, _ = torch.max(pxy*pxy, dim=0) # ny
        #col_my, _ = torch.max(pyx*pyx, dim=0)
        #ratiox = col_sumx / col_mx # ny
        #ratioy = col_sumy / col_my
        #ratiox[ratiox<threshold] = 0.0
        #ratioy[ratioy<threshold] = 0.0
        #lossx = torch.sum(ratiox)
        #lossy = torch.sum(ratioy) #v5
        pxy2 = pxy*pxy
        pyx2 = pyx*pyx
        col_mx, _ = torch.max(pxy2, dim=0, keepdim=True) # ny
        col_my, _ = torch.max(pyx2, dim=0, keepdim=True) # nx
        ratiox = pxy2 / col_mx
        ratioy = pyx2 / col_my
        ratiox[ratiox<threshold] = 0.0
        ratioy[ratioy<threshold] = 0.0
        maskx = col_mx < 0.3
        masky = col_my < 0.3
        ratiox[:, maskx.squeeze(0)] = 0.0
        ratioy[:, masky.squeeze(0)] = 0.0
        lossx = torch.sum(ratiox)
        lossy = torch.sum(ratioy)
        #row_mx, _ = torch.max(pxy, dim=1, keepdim=True)
        #row_my, _ = torch.max(pyx, dim=1, keepdim=True)
        #ratiox = (pxy*pxy) / row_mx
        #ratioy = (pyx*pyx) / row_my
        #col_sumx = torch.sum(ratiox, dim=0)
        #col_sumy = torch.sum(ratioy, dim=0)
        #col_mx, _ = torch.max(ratiox, dim=0)
        #col_my, _ = torch.max(ratioy, dim=0)
        #ratioxx = col_sumx / col_mx
        #ratioyy = col_sumy / col_my
        #ratioxx[ratioxx<threshold] = 0.0
        #ratioyy[ratioyy<threshold] = 0.0
        #lossx = torch.sum(ratioxx)
        #lossy = torch.sum(ratioyy) #v6
        if epoch is None:
            return self.loss_weight * (lossx + lossy)
        elif epoch<8:
            return 0.0 * (lossx + lossy)
        else:
            return self.loss_weight * (lossx + lossy)
