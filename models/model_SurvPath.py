"""
model_SurvPath.py — SurvPath + Bidirectional Gate + SADA
                    (Survival-Aware Disentangled Alignment)

创新点1：Attention Pooling 双向门控（BidirectionalGate）
        在 cross-attention 之前自适应筛选 pathway 和 WSI 信息

创新点2：SADA（Survival-Aware Disentangled Alignment）—— 为生存预测量身定做的对齐模块
        在原 DCA 基础上做了三个针对性升级：
        
        (a) 分解式投影头：shared / specific 双子空间
            灵感：ConFEDE (ACL 2023)
        
        (b) Shared ⊥ Specific 正交约束
            强制 shared 与 specific 子空间正交，防止两者学到重复信息
            （否则分解形同虚设）
        
        
        (c) Shared-Only 辅助生存预测头
            shared 子空间单独接一个生存预测头，强迫 shared 学到风险相关的
            跨模态共享语义，把对齐目标和生存任务真正绑定
        
        (d) Survival-Bin-Guided Contrastive（在 sada_loss.py 中实现）
            生存bin 相近的患者作为软正样本，bin 远的作为硬负样本
            异质性高的数据（如 HNSC）可以利用亚群结构而不是抹平它

架构流程：
    pathway tokens + WSI tokens
        ↓ BidirectionalGate（Attention Pooling 门控）           [创新点1]
    门控后 tokens → cross_attention → mean pooling
        ↓
    paths_postSA_embed, wsi_postSA_embed   (each [B, 128])
        ├─→ to_logits（主预测头）           → logits          (主任务)
        ↓
    SADA module:                                              [创新点2]
        ├─→ shared_projector(两模态共享)    → paths_shared, wsi_shared
        ├─→ specific_projector(两模态各一)  → paths_specific, wsi_specific
        ├─→ ortho_loss = <shared, specific>^2                (a) 正交
        └─→ shared_survival_head([p_sh, w_sh]) → aux_logits  (b) 辅助
    
    在 sada_loss.py 中：
        shared_contrast_loss   (c) 生存bin引导 InfoNCE
        specific_diverge_loss  (d) specific 子空间互相推远
"""

import torch
import os
from models.graph_enhancer import GraphEnhancer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from einops import reduce
from torch.nn import ReLU
from models.layers.cross_attention import FeedForward, MMAttentionLayer
import math
import pandas as pd


def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


# ============================================================
# 双向门控模块（Attention Pooling版）—— 创新点1（保持不变）
# ============================================================

class BidirectionalGate(nn.Module):
    """
    双向门控，使用可学习的 Attention Pooling 生成全局向量。
    """
    def __init__(self, dim):
        super().__init__()
        self.path_attn_score = nn.Linear(dim, 1)
        self.wsi_attn_score  = nn.Linear(dim, 1)

        self.path_to_wsi_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.wsi_to_path_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, paths_tokens, wsi_tokens):
        # Attention Pooling 生成全局向量
        path_score   = self.path_attn_score(paths_tokens)
        path_weight  = F.softmax(path_score, dim=1)
        paths_global = torch.sum(path_weight * paths_tokens, dim=1)

        wsi_score   = self.wsi_attn_score(wsi_tokens)
        wsi_weight  = F.softmax(wsi_score, dim=1)
        wsi_global  = torch.sum(wsi_weight * wsi_tokens, dim=1)

        # 双向门控
        gate_p2w  = self.path_to_wsi_gate(paths_global)
        wsi_gated = wsi_tokens * gate_p2w.unsqueeze(1)

        gate_w2p    = self.wsi_to_path_gate(wsi_global)
        paths_gated = paths_tokens * gate_w2p.unsqueeze(1)

        return paths_gated, wsi_gated


# ============================================================
# SADA: Survival-Aware Disentangled Alignment —— 创新点2
# ============================================================

class SADAProjection(nn.Module):
    """
    Survival-Aware Disentangled Alignment Projection Module.
    
    包含三个组件：
      1. shared_projector       —— 两模态共享参数（学模态不变特征）
      2. specific_projector × 2 —— 每个模态独立（学模态特异特征）
      3. shared_survival_head   —— 用 shared 子空间预测生存（辅助任务）
    
    并直接在 forward 内部计算正交损失（shared ⊥ specific）。
    """
    def __init__(self, input_dim, shared_dim=64, specific_dim=64,
                 num_classes=4, dropout=0.1):
        super().__init__()

        assert shared_dim == specific_dim, \
            "shared_dim 必须等于 specific_dim 才能计算正交内积"

        # ---- 共享投影头（两模态共享参数）----
        self.shared_projector = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
        )

        # ---- 专属投影头（每个模态独立）----
        self.path_specific_projector = nn.Sequential(
            nn.Linear(input_dim, specific_dim),
            nn.LayerNorm(specific_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(specific_dim, specific_dim),
        )
        self.wsi_specific_projector = nn.Sequential(
            nn.Linear(input_dim, specific_dim),
            nn.LayerNorm(specific_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(specific_dim, specific_dim),
        )

        # ---- Shared-Only 辅助生存预测头 ----
        # 输入：[paths_shared ; wsi_shared] 共 2*shared_dim
        # 输出：和主任务一样的 num_classes 个 hazard logits
        # 作用：把"对齐"和"生存预测"绑定起来，强迫 shared 含风险信息
        self.shared_survival_head = nn.Sequential(
            nn.Linear(2 * shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

    def forward(self, paths_embed, wsi_embed):
        """
        Args:
            paths_embed : [B, input_dim]
            wsi_embed   : [B, input_dim]
        Returns:
            dict containing:
                paths_shared, wsi_shared, paths_specific, wsi_specific
                aux_logits  : [B, num_classes]  辅助生存预测
                ortho_loss  : scalar            shared⊥specific 正交损失
        """
        # ---- 分解 ----
        paths_shared = self.shared_projector(paths_embed)
        wsi_shared   = self.shared_projector(wsi_embed)

        paths_specific = self.path_specific_projector(paths_embed)
        wsi_specific   = self.wsi_specific_projector(wsi_embed)

        # ---- 正交损失（shared ⊥ specific 在每个模态内）----
        # 内积平方均值：完全正交时 = 0，方向一致时 = 大
        ortho_loss_path = torch.mean(
            torch.sum(paths_shared * paths_specific, dim=1) ** 2
        )
        ortho_loss_wsi  = torch.mean(
            torch.sum(wsi_shared * wsi_specific, dim=1) ** 2
        )
        ortho_loss = ortho_loss_path + ortho_loss_wsi

        # ---- 辅助生存预测（仅用 shared 子空间）----
        shared_concat = torch.cat([paths_shared, wsi_shared], dim=1)
        aux_logits = self.shared_survival_head(shared_concat)

        return {
            'paths_shared':   paths_shared,
            'wsi_shared':     wsi_shared,
            'paths_specific': paths_specific,
            'wsi_specific':   wsi_specific,
            'aux_logits':     aux_logits,
            'ortho_loss':     ortho_loss,
        }


# ============================================================
# 主模型：SurvPath + BidirectionalGate + SADA
# ============================================================

class SurvPath(nn.Module):
    def __init__(
        self,
        omic_sizes=[100, 200, 300, 400, 500, 600],
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names=[],
        # ========== SADA 参数 ==========
        use_sada=True,            # 是否启用 SADA（保留 use_dca 同名兼容旧 config）
        sada_shared_dim=64,
        sada_specific_dim=64,
        # 旧名兼容（如果你的 config 还在传 use_dca）
        use_dca=None,
        dca_shared_dim=None,
        dca_specific_dim=None,
    ):
        super(SurvPath, self).__init__()

        # 兼容旧参数名
        if use_dca is not None:        use_sada         = use_dca
        if dca_shared_dim is not None: sada_shared_dim   = dca_shared_dim
        if dca_specific_dim is not None: sada_specific_dim = dca_specific_dim

        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim
        self.use_sada = use_sada
        self.num_classes = num_classes

        # omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            self.all_gene_names = list(all_gene_names)

        # WSI projection
        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        # Pathway SNN 网络
        self.init_per_path_model(omic_sizes)

        # [创新点1] 双向门控
        self.bigate = BidirectionalGate(dim=self.wsi_projection_dim)

        # ====== Cross Attention（加深为 2 层 stack） ======
        # 创新点1.5：层次化多模态融合（Hierarchical Cross-Attention）
        #   - 第一层做粗粒度模态融合
        #   - 中间用 LayerNorm + Linear 升维（128 → 256）
        #   - 第二层做精细化模态融合
        self.identity = nn.Identity()
        self.cross_attender_1 = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_pathways
        )
        # 中间升维：第一层输出 128 维 → 升回 256 维供第二层 cross-attention 使用
        self.between_ca_layers = nn.Sequential(
            nn.LayerNorm(self.wsi_projection_dim // 2),
            nn.Linear(self.wsi_projection_dim // 2, self.wsi_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attender_2 = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_pathways
        )

        # FeedForward + LayerNorm
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # [创新点2] SADA 模块
        if self.use_sada:
            self.sada = SADAProjection(
                input_dim    = self.wsi_projection_dim // 2,   # 128
                shared_dim   = sada_shared_dim,
                specific_dim = sada_specific_dim,
                num_classes  = num_classes,
                dropout      = dropout,
            )

        # 主预测头
        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim / 4), self.num_classes)
        )

        self.graph_enabled = os.environ.get('SADA_GRAPH_ENABLED', '1') == '1'
        # REGION_CONTEXT_V1: preserve the historical modes when not selected.
        self.wsi_mode = os.environ.get('SADA_WSI_MODE', 'hypergraph')
        if self.wsi_mode == 'region_context':
            if not self.graph_enabled:
                raise ValueError('region_context requires graph-enabled adapter')
            from models.region_context import RegionContextEnhancer
            self.graph_enhancer = RegionContextEnhancer(self.wsi_projection_dim)
        elif self.wsi_mode == 'mrepath_sheaf':
            from models.mrepath_sheaf import MRePathSheafEncoder
            self.graph_enhancer = MRePathSheafEncoder(self.wsi_projection_dim)
        elif self.wsi_mode == 'cluster_sample':
            if self.graph_enabled:
                raise ValueError('Cluster sampling requires graph disabled')
            self.graph_enhancer = None
        elif self.wsi_mode == 'hypergraph':
            self.graph_enhancer = GraphEnhancer(self.wsi_projection_dim) if self.graph_enabled else None
        else:
            raise ValueError('Unknown SADA_WSI_MODE: ' + self.wsi_mode)

    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

    def forward(self, **kwargs):
        """
        kwargs:
            x_path        : WSI patch features  [1, num_patches, 1024]
            x_omic1..N    : Pathway 基因特征
            return_attn   : 是否返回 attention（可视化）
            return_embed  : 是否返回 SADA 的子空间 embedding 和辅助 logits
                            （训练时必须 True，验证/测试时可以 False 走原路径）
        """
        wsi    = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]
        return_attn  = kwargs.get("return_attn",  False)
        return_embed = kwargs.get("return_embed", False)

        # ---- Step 1: pathway embeddings ----
        h_omic = [self.sig_networks[idx].forward(sig_feat.float())
                  for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)   # [1, num_pathways, 256]

        # ---- Step 2: WSI projection ----
        wsi_embed = self.wsi_projection_net(wsi)         # [1, num_patches, 256]

        if self.graph_enhancer is not None:
            wsi_embed = self.graph_enhancer(wsi_embed, kwargs.get('graph'))

        # [创新点1] Step 3: Attention Pooling 双向门控
        h_omic_gated, wsi_embed_gated = self.bigate(h_omic_bag, wsi_embed)

        # ---- Step 4: cat → cross-attention ----
        tokens = torch.cat([h_omic_gated, wsi_embed_gated], dim=1)
        tokens = self.identity(tokens)

        # ====== 双层 cross-attention ======
        # 第一层（粗融合）
        mm_embed_1 = self.cross_attender_1(x=tokens, mask=None, return_attention=False)
        # 升维 128 → 256，供第二层使用
        mm_embed_1_up = self.between_ca_layers(mm_embed_1)
        # 第二层（精融合）—— 只对它返回 attention（最后一层最有意义）
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = \
                self.cross_attender_2(x=mm_embed_1_up, mask=None, return_attention=True)
        else:
            mm_embed = self.cross_attender_2(x=mm_embed_1_up, mask=None, return_attention=False)

        # ---- Step 5: FeedForward + LayerNorm ----
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        # ---- Step 6: mean pooling ----
        paths_postSA_embed = torch.mean(mm_embed[:, :self.num_pathways, :], dim=1)  # [B, 128]
        wsi_postSA_embed   = torch.mean(mm_embed[:, self.num_pathways:, :], dim=1)  # [B, 128]

        # ---- Step 7: 主预测头 ----
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)        # [B, 256]
        logits = self.to_logits(embedding)

        # [创新点2] Step 8: SADA 分解 + 辅助任务 + 正交损失
        sada_outputs = None
        if self.use_sada and return_embed:
            sada_outputs = self.sada(paths_postSA_embed, wsi_postSA_embed)

        # ---- 返回值兼容性处理 ----
        if return_embed:
            out = {'logits': logits}
            if sada_outputs is not None:
                out.update(sada_outputs)
            if return_attn:
                out['attn_pathways']        = attn_pathways
                out['cross_attn_pathways']  = cross_attn_pathways
                out['cross_attn_histology'] = cross_attn_histology
            return out

        if return_attn:
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits

    # ----------- captum 接口（保持原签名，确保归因分析不破）-----------
    def captum(self, omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329 ,omics_330, wsi, graph=None):

        omic_list = [omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329]

        h_omic = [self.sig_networks[idx].forward(sig_feat)
                  for idx, sig_feat in enumerate(omic_list)]
        h_omic_bag = torch.stack(h_omic, dim=1)

        wsi_embed = self.wsi_projection_net(wsi)
        if self.graph_enhancer is not None:
            wsi_embed = self.graph_enhancer(wsi_embed, graph)
        h_omic_gated, wsi_embed_gated = self.bigate(h_omic_bag, wsi_embed)

        tokens = torch.cat([h_omic_gated, wsi_embed_gated], dim=1)
        tokens = self.identity(tokens)
        mm_embed_1 = self.cross_attender_1(x=tokens, mask=None, return_attention=False)
        mm_embed_1_up = self.between_ca_layers(mm_embed_1)
        mm_embed = self.cross_attender_2(x=mm_embed_1_up, mask=None, return_attention=False)
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        paths_postSA_embed = torch.mean(mm_embed[:, :self.num_pathways, :], dim=1)
        wsi_postSA_embed   = torch.mean(mm_embed[:, self.num_pathways:, :], dim=1)
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

        logits = self.to_logits(embedding)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        return risk

