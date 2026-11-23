"""
sada_loss.py — SADA 模块的损失函数集合

包含两部分（配合 model_SurvPath.py 中的 SADAProjection 使用）：

  1. survival_bin_guided_contrastive_loss
     生存 bin 引导的跨模态 InfoNCE 损失
     —— 在 shared 子空间中将「生存bin相近的样本」当作正样本族群
     这样异质性高的数据（如 HNSC、COADREAD）可以利用亚群结构而不是抹平它

  2. specific_diverge_loss
     specific 子空间互相推远（diverge）损失
     —— 防止 specific 子空间也学到模态共享信息

正交损失（ortho_loss）已经在 SADAProjection.forward 中直接返回，
辅助生存损失（aux survival NLL）请直接复用你已有的 NLLSurvLoss / CoxLoss。

================================================================
总损失公式（在训练循环中组合）：
    L_total = L_main_surv                       # 主任务 NLL/CoxSurv
            + λ_aux  * L_aux_surv                # shared-only 辅助生存损失
            + λ_ortho * L_ortho                  # 正交（来自模型 forward）
            + λ_share * L_shared_contrast        # 生存bin引导对比
            + λ_spec  * L_specific_diverge       # specific 推远

建议初始超参（lambda）：
    λ_aux   = 0.3
    λ_ortho = 0.1
    λ_share = 0.5
    λ_spec  = 0.1
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 生存 bin 引导对比损失（核心创新）
# ============================================================

def survival_bin_guided_contrastive_loss(
    paths_shared,         # [B, D]   shared 子空间中的 pathway 表示
    wsi_shared,           # [B, D]   shared 子空间中的 WSI     表示
    survival_bins,        # [B]      每个样本的离散生存 bin (long tensor)
    temperature=0.1,
    bin_threshold=1,      # 生存bin距离 <= 该值视为正样本族群（软正样本）
):
    """
    生存bin引导的跨模态 InfoNCE：

    对于 anchor paths_shared[i]：
      - wsi_shared[j] 中所有 |bin_j - bin_i| <= threshold 的 j 视为正样本
      - 其余视为负样本
    对称地，对 wsi_shared[i] 同样处理。

    关键性质：
      - 同一样本（i=i）的两模态必为正（对角线）
      - 生存预后相近的不同样本也作为正样本族群
      - 异质群体内反而强化了风险层级，而不是被压平

    如果 batch_size < 2，返回 0（无法做对比）。
    """
    B = paths_shared.size(0)
    device = paths_shared.device

    if B < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # L2 normalize（InfoNCE 标准做法）
    p_sh = F.normalize(paths_shared, dim=-1)
    w_sh = F.normalize(wsi_shared,   dim=-1)

    # 相似度矩阵：paths anchor 与 所有 wsi 之间
    sim_p2w = torch.matmul(p_sh, w_sh.T) / temperature   # [B, B]
    sim_w2p = sim_p2w.T                                  # [B, B]

    # 生存 bin 距离矩阵
    bins = survival_bins.view(-1, 1).long()              # [B, 1]
    bin_dist = torch.abs(bins - bins.T).float()          # [B, B]

    # 正样本 mask：bin 距离 <= threshold
    pos_mask = (bin_dist <= bin_threshold).float()       # [B, B]
    # 对角线（同样本两模态）必为正
    pos_mask.fill_diagonal_(1.0)

    # ---- p→w 方向 ----
    # 数值稳定的 log-sum-exp 写法
    max_p2w = sim_p2w.max(dim=1, keepdim=True)[0].detach()
    exp_p2w = torch.exp(sim_p2w - max_p2w)               # [B, B]
    denom_p = exp_p2w.sum(dim=1)                          # [B]
    numer_p = (exp_p2w * pos_mask).sum(dim=1)             # [B]
    loss_p2w = -torch.log(numer_p / (denom_p + 1e-8) + 1e-8)

    # ---- w→p 方向（对称）----
    max_w2p = sim_w2p.max(dim=1, keepdim=True)[0].detach()
    exp_w2p = torch.exp(sim_w2p - max_w2p)
    denom_w = exp_w2p.sum(dim=1)
    numer_w = (exp_w2p * pos_mask).sum(dim=1)
    loss_w2p = -torch.log(numer_w / (denom_w + 1e-8) + 1e-8)

    return 0.5 * (loss_p2w.mean() + loss_w2p.mean())


# ============================================================
# 2. specific 子空间推远（diverge）损失
# ============================================================

def specific_diverge_loss(paths_specific, wsi_specific):
    """
    希望 paths_specific 和 wsi_specific 互相不同（保留模态差异）。
    
    做法：直接最小化它们的余弦相似度（即推远）。
    """
    p_sp = F.normalize(paths_specific, dim=-1)
    w_sp = F.normalize(wsi_specific,   dim=-1)
    # 同一样本的两模态 specific 余弦相似度
    cos_sim = (p_sp * w_sp).sum(dim=-1)   # [B]
    # 余弦相似度越接近 0 越好（互相正交即互相独立）
    # 用平方让 loss 始终非负
    return (cos_sim ** 2).mean()


# ============================================================
# 一键计算 SADA 全部辅助损失的便捷封装
# ============================================================

class SADALoss(nn.Module):
    """
    一键计算 SADA 的所有辅助损失。
    主任务 survival loss 仍然由调用方自己算（保持你原有 NLL/Cox 逻辑）。

    用法：
        sada_criterion = SADALoss(
            lambda_aux=0.3, lambda_ortho=0.1,
            lambda_share=0.5, lambda_spec=0.1,
            temperature=0.1, bin_threshold=1,
        )

        out = model(**batch, return_embed=True)
        # 主任务损失（你自己算）
        loss_main = nll_surv_loss(out['logits'], y, c)
        # 辅助任务损失（用 shared 子空间预测生存）
        loss_aux  = nll_surv_loss(out['aux_logits'], y, c)
        # 一键计算 SADA 其余损失
        sada_losses = sada_criterion(
            out, survival_bins=y,
            loss_main=loss_main, loss_aux=loss_aux,
        )
        loss_total = sada_losses['loss_total']
        loss_total.backward()
    """
    def __init__(
        self,
        lambda_aux=0.3,
        lambda_ortho=0.1,
        lambda_share=0.5,
        lambda_spec=0.1,
        temperature=0.1,
        bin_threshold=1,
    ):
        super().__init__()
        self.lambda_aux   = lambda_aux
        self.lambda_ortho = lambda_ortho
        self.lambda_share = lambda_share
        self.lambda_spec  = lambda_spec
        self.temperature  = temperature
        self.bin_threshold = bin_threshold

    def forward(self, sada_out, survival_bins, loss_main, loss_aux=None):
        """
        Args:
            sada_out      : 模型 forward(return_embed=True) 的返回 dict
            survival_bins : [B] long tensor，每个样本的离散生存 bin
            loss_main     : 主任务 survival loss（scalar tensor）
            loss_aux      : shared-only 辅助 survival loss
                           （None 时由调用方决定是否启用 aux）
        Returns:
            dict {
              'loss_total'   : 总损失（用它 backward）
              'loss_main'    : 主任务
              'loss_aux'     : 辅助生存
              'loss_ortho'   : 正交
              'loss_share'   : 生存bin对比
              'loss_spec'    : specific 推远
            }
        """
        # 1) 正交（来自模型 forward）
        loss_ortho = sada_out['ortho_loss']

        # 2) 生存bin对比（在 shared 子空间）
        loss_share = survival_bin_guided_contrastive_loss(
            sada_out['paths_shared'],
            sada_out['wsi_shared'],
            survival_bins,
            temperature=self.temperature,
            bin_threshold=self.bin_threshold,
        )

        # 3) specific 推远
        loss_spec = specific_diverge_loss(
            sada_out['paths_specific'],
            sada_out['wsi_specific'],
        )

        # 4) 总损失
        loss_total = (
            loss_main
            + self.lambda_ortho * loss_ortho
            + self.lambda_share * loss_share
            + self.lambda_spec  * loss_spec
        )
        if loss_aux is not None:
            loss_total = loss_total + self.lambda_aux * loss_aux

        return {
            'loss_total': loss_total,
            'loss_main':  loss_main.detach(),
            'loss_aux':   loss_aux.detach()   if loss_aux is not None else None,
            'loss_ortho': loss_ortho.detach(),
            'loss_share': loss_share.detach() if isinstance(loss_share, torch.Tensor) else loss_share,
            'loss_spec':  loss_spec.detach(),
        }