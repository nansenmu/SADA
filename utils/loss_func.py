import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import pdb


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    NLL loss for discrete time-to-event modeling (Zadeh and Schmid, 2020).
    """
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


# ============================================================
# [创新点2] 分解式跨模态对比对齐损失
# ============================================================

class DecomposedContrastiveLoss(nn.Module):
    """
    分解式跨模态对比对齐损失（Decomposed Contrastive Alignment Loss, DCA Loss）

    灵感来源：
      [1] Yang et al., "ConFEDE: Contrastive Feature Decomposition for Multimodal
          Sentiment Analysis," ACL 2023
      [2] Kang et al., "CLGSI: Contrastive Learning Guided by Sentiment Intensity 
          for MSA," NAACL Findings 2024
      [3] Hazarika et al., "MISA: Modality-Invariant and -Specific Representations
          for Multimodal Sentiment Analysis," ACM MM 2020

    在情感分析中，同一视频片段的文本/语音/视觉三个模态都共同反映情感强度，
    但又各自保留模态特异的情感线索（面部表情、语调、语义）。
    本任务中，WSI形态学和Pathway基因表达同样共同反映肿瘤预后信号，
    但各自保留形态特异（组织结构）和分子特异（基因通路）的信息。
    
    三个子损失：
      L_align  : 共享子空间InfoNCE对齐（缩小模态gap）
      L_diverse: 专属子空间正交化（保留模态独特性，防止过度对齐）
      L_supcon : 生存风险引导的监督对比（hard negative mining，利用标签信息）

    Args:
        temperature    : InfoNCE 温度系数，越小对比越"尖锐"，常用0.07~0.2
        lambda_align   : L_align   权重，建议 0.1~1.0
        lambda_diverse : L_diverse 权重，建议 0.05~0.5
        lambda_supcon  : L_supcon  权重，建议 0.0~0.5，无标签或小batch可关
        use_supcon     : 是否启用监督对比（需要生存风险标签，且buffer>1时才有意义）
    """
    def __init__(
        self,
        temperature    = 0.1,
        lambda_align   = 0.5,
        lambda_diverse = 0.1,
        lambda_supcon  = 0.2,
        use_supcon     = True,
    ):
        super().__init__()
        self.temperature    = temperature
        self.lambda_align   = lambda_align
        self.lambda_diverse = lambda_diverse
        self.lambda_supcon  = lambda_supcon
        self.use_supcon     = use_supcon

    def forward(
        self,
        paths_shared,        # [B, D_shared]
        wsi_shared,          # [B, D_shared]
        paths_specific,      # [B, D_specific]
        wsi_specific,        # [B, D_specific]
        risk_labels=None,    # [B]  生存风险分组 (0=低风险, 1=高风险)，可为 None
    ):
        """
        Returns:
            loss_total : 标量 tensor
            loss_dict  : dict，便于训练时日志打印三项子损失
        """
        B = paths_shared.shape[0]
        loss_dict = {}

        # B=1 时对比损失没有意义，返回 0
        if B < 2:
            zero = paths_shared.new_zeros(())
            return zero, {'align': 0.0, 'diverse': 0.0, 'supcon': 0.0}

        # ---- L2 归一化（对比学习的标准操作）----
        ps = F.normalize(paths_shared,   dim=-1)
        ws = F.normalize(wsi_shared,     dim=-1)
        pd_ = F.normalize(paths_specific, dim=-1)
        wd = F.normalize(wsi_specific,   dim=-1)

        # ============================================================
        # (1) L_align  : 共享子空间 InfoNCE 双向对齐
        #     同一患者的 (paths_shared, wsi_shared) 为正样本对
        #     batch 内其他患者为负样本
        # ============================================================
        sim_pw = torch.mm(ps, ws.t()) / self.temperature   # [B, B]
        sim_wp = sim_pw.t()                                # [B, B]
        labels = torch.arange(B, device=ps.device)
        loss_align = 0.5 * (
            F.cross_entropy(sim_pw, labels) +
            F.cross_entropy(sim_wp, labels)
        )
        loss_dict['align'] = loss_align.item()

        # ============================================================
        # (2) L_diverse : 专属子空间差异性损失
        #     强制同一患者的 paths_specific 和 wsi_specific 不相似
        #     防止 DCA 退化为"两个模态都学到相同表示"
        #
        #     这里用余弦相似度的平方（稳定且可求导）
        # ============================================================
        cos_sim_specific = (pd_ * wd).sum(dim=-1)          # [B]
        loss_diverse = (cos_sim_specific ** 2).mean()
        loss_dict['diverse'] = loss_diverse.item()

        # ============================================================
        # (3) L_supcon  : 生存风险引导的监督对比（可选）
        #     同风险组的跨模态对作为额外正样本（扩大正样本集合）
        #     基于 CLGSI (NAACL 2024) 的标签引导采样思想
        #
        #     【鲁棒实现】防御以下nan来源：
        #       a) buffer内所有样本属于同一风险组 → 无负样本 → log_prob为-inf
        #       b) masked_fill(-inf) 后 logsumexp 可能为 -inf → 减法得 nan
        #       c) 标签shape异常（多维tensor）
        # ============================================================
        loss_supcon = paths_shared.new_zeros(())
        if self.use_supcon and risk_labels is not None and self.lambda_supcon > 0:
            # 强制 1D long tensor
            risk_labels = risk_labels.reshape(-1).long()

            # 防御 a：整个 buffer 只有一个风险组 → 跳过 sup 损失
            if len(torch.unique(risk_labels)) >= 2 and risk_labels.shape[0] == B:
                # 堆叠 pathway + WSI 共享特征 [2B, D]
                feats = torch.cat([ps, ws], dim=0)                          # [2B, D]
                sup_labels = torch.cat([risk_labels, risk_labels], dim=0)   # [2B]

                sim_matrix = torch.mm(feats, feats.t()) / self.temperature  # [2B, 2B]

                # 对角线（自己 vs 自己）mask
                eye = torch.eye(2 * B, device=feats.device, dtype=torch.bool)

                # 正样本 mask：同风险组 & 排除自身
                sup_mask = (sup_labels.unsqueeze(0) == sup_labels.unsqueeze(1)).float()
                sup_mask = sup_mask.masked_fill(eye, 0)

                # 负样本 mask：任何非自身的样本（含同组和异组）
                neg_valid = (~eye).float()

                # 数值稳定的 logsumexp：
                # 先把 sim_matrix 对角位置置成很小的数（不是 -inf，避免 logsumexp 出 -inf）
                sim_for_lse = sim_matrix.masked_fill(eye, -1e4)
                log_prob = sim_matrix - torch.logsumexp(sim_for_lse, dim=1, keepdim=True)

                # 只对"有正样本的行"做平均
                pos_count = sup_mask.sum(dim=1)                 # [2B]
                has_pos = pos_count > 0
                mean_log_prob_pos = (sup_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

                # 过滤掉 nan/inf 行
                valid = has_pos & torch.isfinite(mean_log_prob_pos)
                if valid.any():
                    loss_supcon = -mean_log_prob_pos[valid].mean()

        # 防御 b/c：最后再做一次兜底，避免 nan 污染总 loss
        if not torch.isfinite(loss_supcon):
            loss_supcon = paths_shared.new_zeros(())

        loss_dict['supcon'] = float(loss_supcon.item()) if isinstance(loss_supcon, torch.Tensor) else 0.0

        # ============================================================
        # 总损失
        # ============================================================
        loss_total = (
            self.lambda_align   * loss_align   +
            self.lambda_diverse * loss_diverse +
            self.lambda_supcon  * loss_supcon
        )
        return loss_total, loss_dict