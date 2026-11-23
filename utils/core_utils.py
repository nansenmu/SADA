from ast import Lambda
import numpy as np
import pdb
import os
import pickle
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics
# DUAL_BRANCH_V2: default preserves the previous region/hypergraph versions.
if os.environ.get('SADA_MODEL_VARIANT', 'single') == 'joint_hypergraph':
    from models.model_joint_hypergraph import SurvPath
elif os.environ.get('SADA_MODEL_VARIANT', 'single') == 'dual_branch':
    from models.model_dual_branch import SurvPath
elif os.environ.get('SADA_MODEL_VARIANT', 'single') == 'attention_v3':
    from models.model_attention_v3 import SurvPath
elif os.environ.get('SADA_MODEL_VARIANT', 'single') == 'single':
    from models.model_SurvPath import SurvPath
else:
    raise ValueError('Unknown SADA_MODEL_VARIANT')
from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
from models.model_TMIL import TMIL
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)


import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss
from utils.sada_loss import (
    survival_bin_guided_contrastive_loss,
    specific_diverge_loss,
)

import torch.optim as optim


SADA_ENABLED       = os.environ.get('SADA_ENABLED', '1') == '1'
SADA_BUFFER_SIZE   = int(os.environ.get('SADA_BUFFER_SIZE',   '8'))
SADA_TEMPERATURE   = float(os.environ.get('SADA_TEMP',        '0.1'))
SADA_BIN_THRESHOLD = int(os.environ.get('SADA_BIN_THRESHOLD', '1'))
SADA_LAMBDA_AUX    = float(os.environ.get('SADA_LAMBDA_AUX',   '0.3'))
SADA_LAMBDA_ORTHO  = float(os.environ.get('SADA_LAMBDA_ORTHO', '0.1'))
SADA_LAMBDA_SHARE  = float(os.environ.get('SADA_LAMBDA_SHARE', '0.5'))
SADA_LAMBDA_SPEC   = float(os.environ.get('SADA_LAMBDA_SPEC',  '0.1'))
SADA_WARMUP_EPOCHS = int(os.environ.get('SADA_WARMUP_EPOCHS',  '2'))

print("\n[SADA env] enabled={}, buffer={}, T={}, bin_thresh={}, "
      "lambda(aux/ortho/share/spec)={}/{}/{}/{}, warmup={}".format(
    SADA_ENABLED, SADA_BUFFER_SIZE, SADA_TEMPERATURE, SADA_BIN_THRESHOLD,
    SADA_LAMBDA_AUX, SADA_LAMBDA_ORTHO, SADA_LAMBDA_SHARE, SADA_LAMBDA_SPEC,
    SADA_WARMUP_EPOCHS))


def _get_splits(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    return train_split, val_split


def _init_loss_function(args):
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn


def _init_optim(args, model):
    print('\nInit optimizer ...', end=' ')
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def _init_model(args):
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0

    if args.modality == "mlp_per_path":
        model_dict = {"device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes}
        model = MaskedOmics(**model_dict)
    elif args.modality == "omics":
        model_dict = {"input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout}
        model = MLPOmics(**model_dict)
    elif args.modality == "snn":
        model_dict = {"omic_input_dim" : omics_input_dim}
        model = SNNOmics(**model_dict)
    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:
        model_dict = {"device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim, "fusion":args.fusion}
        model = ABMIL(**model_dict)
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:
        model_dict = {"device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim, "fusion":args.fusion}
        model = DeepMISL(**model_dict)
    elif args.modality == "mlp_wsi":
        model_dict = {"wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim,
            "dropout":args.encoder_dropout, "device": args.device}
        model = MLPWSI(**model_dict)
    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:
        model_dict = {"device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim, "fusion":args.fusion}
        model = TMIL(**model_dict)
    elif args.modality == "survpath":
        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}
        model_dict['use_dca'] = SADA_ENABLED
        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))
    print('Done!')
    _print_network(args.results_dir, model)
    return model


def _init_loaders(args, train_split, val_split):
    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None
    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')
    return train_loader, val_loader


def _extract_survival_metadata(train_loader, val_loader):
    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()], axis=0)
    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()], axis=0)
    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival


def _unpack_data(modality, device, data):
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]; mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device); data_omics = data[1].to(device)
        mask = None if data[6][0,0] == 1 else data[6].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    elif modality in ["coattn", "coattn_motcat"]:
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]
        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)
    elif modality in ["survpath"]:
        data_WSI = data[0].to(device)
        data_omics = [item.to(device) for item in data[1][0]]
        mask = data[6].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    else:
        raise ValueError('Unsupported modality:', modality)
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)
    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask


def _process_data_and_forward(model, modality, device, data, return_embed=False):
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
    if modality in ["coattn", "coattn_motcat"]:
        out = model(x_path=data_WSI, x_omic1=data_omics[0], x_omic2=data_omics[1], x_omic3=data_omics[2],
            x_omic4=data_omics[3], x_omic5=data_omics[4], x_omic6=data_omics[5])
    elif modality == 'survpath':
        input_args = {"x_path": data_WSI.to(device)}
        input_args["graph"] = mask
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        input_args["return_embed"] = return_embed
        out = model(**input_args)
    else:
        out = model(data_omics=data_omics, data_WSI=data_WSI, mask=mask)

    if isinstance(out, dict):
        h = out['logits']
        aux_logits = out.get('aux_logits', None)
        ortho_loss = out.get('ortho_loss', None)
        sada_embeds = {
            'paths_shared':   out.get('paths_shared'),
            'wsi_shared':     out.get('wsi_shared'),
            'paths_specific': out.get('paths_specific'),
            'wsi_specific':   out.get('wsi_specific'),
        }
    else:
        h = out
        aux_logits = None
        ortho_loss = None
        sada_embeds = None

    if len(h.shape) == 1:
        h = h.unsqueeze(0)
    if aux_logits is not None and len(aux_logits.shape) == 1:
        aux_logits = aux_logits.unsqueeze(0)

    return h, aux_logits, ortho_loss, sada_embeds, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()


def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data


def _flush_buffer_and_update(buf, optimizer, scheduler, loss_fn, log_aux_share_spec_only=True):
    if len(buf['paths_shared']) == 0:
        return None
    ps = torch.cat(buf['paths_shared'],   dim=0)
    ws = torch.cat(buf['wsi_shared'],     dim=0)
    pp = torch.cat(buf['paths_specific'], dim=0)
    wp = torch.cat(buf['wsi_specific'],   dim=0)
    bins = torch.cat(buf['survival_bins'], dim=0)
    loss_main_buf  = torch.stack(buf['nll_main']).mean()
    loss_aux_buf   = torch.stack(buf['nll_aux']).mean()  if len(buf['nll_aux'])  > 0 else None
    loss_ortho_buf = torch.stack(buf['ortho']).mean()    if len(buf['ortho'])    > 0 else None
    if ps.shape[0] >= 2:
        loss_share = survival_bin_guided_contrastive_loss(
            ps, ws, bins, temperature=SADA_TEMPERATURE, bin_threshold=SADA_BIN_THRESHOLD)
        loss_spec = specific_diverge_loss(pp, wp)
    else:
        loss_share = torch.tensor(0.0, device=ps.device, requires_grad=True)
        loss_spec  = torch.tensor(0.0, device=ps.device, requires_grad=True)
    loss_total = loss_main_buf
    if loss_aux_buf is not None:
        loss_total = loss_total + SADA_LAMBDA_AUX * loss_aux_buf
    if loss_ortho_buf is not None:
        loss_total = loss_total + SADA_LAMBDA_ORTHO * loss_ortho_buf
    loss_total = loss_total + SADA_LAMBDA_SHARE * loss_share + SADA_LAMBDA_SPEC * loss_spec
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    info = {
        'aux':   loss_aux_buf.item()   if loss_aux_buf   is not None else 0.0,
        'ortho': loss_ortho_buf.item() if loss_ortho_buf is not None else 0.0,
        'share': loss_share.item()     if isinstance(loss_share, torch.Tensor) else float(loss_share),
        'spec':  loss_spec.item()      if isinstance(loss_spec,  torch.Tensor) else float(loss_spec),
    }
    for k in buf:
        buf[k].clear()
    return info


def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss_nll = 0.
    sada_logs = {'aux': 0., 'ortho': 0., 'share': 0., 'spec': 0.}
    sada_update_count = 0
    all_risk_scores = []; all_censorships = []; all_event_times = []; all_clinical_data = []
    use_sada_this_epoch = (SADA_ENABLED and modality == "survpath" and epoch >= SADA_WARMUP_EPOCHS)
    buf = {
        'paths_shared': [], 'wsi_shared': [],
        'paths_specific': [], 'wsi_specific': [],
        'survival_bins': [],
        'nll_main': [], 'nll_aux':  [], 'ortho':    [],
    }
    for batch_idx, data in enumerate(loader):
        h, aux_logits, ortho_loss, sada_embeds, y_disc, event_time, censor, clinical_data_list = \
            _process_data_and_forward(model, modality, device, data, return_embed=use_sada_this_epoch)
        loss_nll = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss_nll = loss_nll / y_disc.shape[0]
        total_loss_nll += loss_nll.item()
        risk, _ = _calculate_risk(h)
        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
            all_risk_scores, all_censorships, all_event_times, all_clinical_data,
            event_time, censor, risk, clinical_data_list)
        if use_sada_this_epoch and sada_embeds is not None and sada_embeds['paths_shared'] is not None:
            loss_aux = loss_fn(h=aux_logits, y=y_disc, t=event_time, c=censor) / y_disc.shape[0]
            buf['paths_shared'].append(sada_embeds['paths_shared'])
            buf['wsi_shared'].append(sada_embeds['wsi_shared'])
            buf['paths_specific'].append(sada_embeds['paths_specific'])
            buf['wsi_specific'].append(sada_embeds['wsi_specific'])
            buf['survival_bins'].append(y_disc.reshape(-1).long())
            buf['nll_main'].append(loss_nll)
            buf['nll_aux'].append(loss_aux)
            if ortho_loss is not None:
                buf['ortho'].append(ortho_loss)
            if len(buf['paths_shared']) >= SADA_BUFFER_SIZE:
                info = _flush_buffer_and_update(buf, optimizer, scheduler, loss_fn)
                if info is not None:
                    for k in sada_logs:
                        sada_logs[k] += info[k]
                    sada_update_count += 1
        else:
            optimizer.zero_grad()
            loss_nll.backward()
            optimizer.step()
            scheduler.step()
        if (batch_idx % 20) == 0:
            if use_sada_this_epoch and sada_update_count > 0:
                print("batch: {}, nll: {:.3f}  | (avg) aux={:.3f}, ortho={:.4f}, share={:.3f}, spec={:.4f}".format(
                    batch_idx, loss_nll.item(),
                    sada_logs['aux']   / max(sada_update_count, 1),
                    sada_logs['ortho'] / max(sada_update_count, 1),
                    sada_logs['share'] / max(sada_update_count, 1),
                    sada_logs['spec']  / max(sada_update_count, 1)))
            else:
                print("batch: {}, nll: {:.3f}".format(batch_idx, loss_nll.item()))
    if use_sada_this_epoch:
        if len(buf['paths_shared']) >= 2:
            info = _flush_buffer_and_update(buf, optimizer, scheduler, loss_fn)
            if info is not None:
                for k in sada_logs:
                    sada_logs[k] += info[k]
                sada_update_count += 1
        elif len(buf['nll_main']) >= 1:
            loss_main_buf  = torch.stack(buf['nll_main']).mean()
            loss_total = loss_main_buf
            if len(buf['nll_aux']) > 0:
                loss_total = loss_total + SADA_LAMBDA_AUX * torch.stack(buf['nll_aux']).mean()
            if len(buf['ortho']) > 0:
                loss_total = loss_total + SADA_LAMBDA_ORTHO * torch.stack(buf['ortho']).mean()
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            for k in buf:
                buf[k].clear()
    total_loss_nll /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    if use_sada_this_epoch and sada_update_count > 0:
        print('Epoch: {}, nll: {:.4f} | aux={:.3f}, ortho={:.4f}, share={:.3f}, spec={:.4f} | train_c_index: {:.4f}'.format(
            epoch, total_loss_nll,
            sada_logs['aux']   / sada_update_count,
            sada_logs['ortho'] / sada_update_count,
            sada_logs['share'] / sada_update_count,
            sada_logs['spec']  / sada_update_count,
            c_index))
    else:
        print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss_nll, c_index))
    return c_index, total_loss_nll


def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        return c_index, c_index_ipcw, BS, IBS, iauc
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        c_index_ipcw = 0.
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        BS = 0.
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        IBS = 0.
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        iauc = 0.
    return c_index, c_index_ipcw, BS, IBS, iauc


def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.
    all_risk_scores = []; all_risk_by_bin_scores = []
    all_censorships = []; all_event_times = []; all_clinical_data = []
    all_logits = []; all_slide_ids = []
    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            if modality in ["coattn", "coattn_motcat"]:
                h = model(x_path=data_WSI, x_omic1=data_omics[0], x_omic2=data_omics[1], x_omic3=data_omics[2],
                    x_omic4=data_omics[3], x_omic5=data_omics[4], x_omic6=data_omics[5])
            elif modality == "survpath":
                input_args = {"x_path": data_WSI.to(device)}
                input_args["graph"] = mask
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                input_args["return_embed"] = True
                out = model(**input_args)
                if isinstance(out, dict) and out.get('aux_logits') is not None:
                    h = 0.7 * out['logits'] + 0.3 * out['aux_logits']
                else:
                    h = out if not isinstance(out, dict) else out['logits']
            else:
                h = model(data_omics=data_omics, data_WSI=data_WSI, mask=mask)
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(
                all_risk_scores, all_censorships, all_event_times, all_clinical_data,
                event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)
    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs
    warmup_steps = warmup_epochs * len(dataloader) if warmup_epochs > 0 else 0
    if scheduler_name == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps)
    elif scheduler_name == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs)
    elif scheduler_name == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs)
    return lr_scheduler


# ============================================================
# ★★ 核心改动：同时保留 best 和 last 两套结果 ★★
# ============================================================
def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    all_survival = _extract_survival_metadata(train_loader, val_loader)

    if SADA_ENABLED and args.modality == "survpath":
        print('\n[SADA] Survival-Aware Disentangled Alignment ENABLED')
        print('[SADA] buffer={}, T={}, bin_thresh={}, lambda(aux/ortho/share/spec)={}/{}/{}/{}'.format(
            SADA_BUFFER_SIZE, SADA_TEMPERATURE, SADA_BIN_THRESHOLD,
            SADA_LAMBDA_AUX, SADA_LAMBDA_ORTHO, SADA_LAMBDA_SHARE, SADA_LAMBDA_SPEC))
        print('[SADA] warmup_epochs={} (first {} epochs train NLL only)\n'.format(
            SADA_WARMUP_EPOCHS, SADA_WARMUP_EPOCHS))

    # ---- 跟踪 best ----
    best_val_cindex = -1.0
    best_metrics = None
    best_results_dict = None
    best_epoch = -1
    best_state_dict = None

    # ---- 跟踪 last（每 epoch 都覆盖）----
    last_metrics = None
    last_results_dict = None
    last_state_dict = None

    # ---- 每 epoch val 历史（写到 json 用）----
    epoch_history = []

    for epoch in range(args.max_epochs):
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)

        results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        print('Epoch:{} Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
            epoch, val_cindex, val_cindex_ipcw, val_IBS, val_iauc))

        epoch_history.append({
            'epoch': epoch,
            'val_cindex': float(val_cindex),
            'val_cindex_ipcw': float(val_cindex_ipcw),
            'val_IBS': float(val_IBS),
            'val_iauc': float(val_iauc),
        })
        # ATTENTION_V3 diagnostics
        if hasattr(getattr(model, 'graph_enhancer', None), 'summary_strength'):
            summary_weight = float(model.graph_enhancer.summary_strength.detach().tanh().cpu())
            local = float(model.graph_enhancer.local_strength.detach().tanh().cpu())
            global_weight = float(model.graph_enhancer.global_strength.detach().tanh().cpu())
            epoch_history[-1].update(summary_residual=summary_weight, local_residual=local, global_residual=global_weight)
            print(f'[AttentionV3] epoch={epoch} summary={summary_weight:.6f} local={local:.6f} global={global_weight:.6f}', flush=True)
        # DUAL_BRANCH_DIAGNOSTICS_V2
        if hasattr(model, 'context_mix_logit'):
            mix = float(model.context_mix_logit.detach().sigmoid().cpu())
            local = float(model.graph_enhancer.local_strength.detach().tanh().cpu())
            global_weight = float(model.graph_enhancer.global_strength.detach().tanh().cpu())
            epoch_history[-1].update(context_mix=mix, local_residual=local, global_residual=global_weight)
            print(f'[DualBranch] epoch={epoch} raw={1-mix:.4f} context={mix:.4f} local={local:.4f} global={global_weight:.4f}', flush=True)

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_metrics = (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)
            best_results_dict = results_dict
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ★ 每 epoch 覆盖 last，最后循环结束时留下的就是 last epoch 的
        last_metrics = (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)
        last_results_dict = results_dict
        last_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- 保存 best epoch 的产物 ----
    if best_state_dict is not None:
        torch.save(best_state_dict, os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # ---- ★ 新增：保存 last epoch 的产物 ----
    if last_state_dict is not None:
        torch.save(last_state_dict, os.path.join(args.results_dir, "s_{}_checkpoint_last.pt".format(cur)))
    if last_results_dict is not None:
        last_pkl_path = os.path.join(args.results_dir, "split_{}_results_last.pkl".format(cur))
        with open(last_pkl_path, 'wb') as f:
            pickle.dump(last_results_dict, f)

    # ---- ★ 新增：把每 epoch val 曲线存到 json ----
    import json
    history_path = os.path.join(args.results_dir, "epoch_history_{}.json".format(cur))
    with open(history_path, 'w') as f:
        json.dump(epoch_history, f, indent=2)

    # ---- 汇总打印 ----
    val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = best_metrics
    val_cindex_last, val_cindex_ipcw_last, val_BS_last, val_IBS_last, val_iauc_last, total_loss_last = last_metrics

    print('=' * 70)
    print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
        val_cindex, val_cindex_ipcw, val_IBS, val_iauc))
    print('Best Val c-index: {:.4f} (at epoch {})'.format(best_val_cindex, best_epoch))
    print('>>>> [LAST EPOCH] Val c-index: {:.4f} | IBS: {:.4f} | iauc: {:.4f} (epoch {})'.format(
        val_cindex_last, val_IBS_last, val_iauc_last, args.max_epochs - 1))
    print('=' * 70)

    # ---- 返回：前两项和以前一样（best 结果 + best metrics）；多返回一个 last_info dict ----
    return best_results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss), {
        'last_val_cindex':      float(val_cindex_last),
        'last_val_cindex_ipcw': float(val_cindex_ipcw_last),
        'last_val_BS':          val_BS_last if isinstance(val_BS_last, (list, np.ndarray)) else float(val_BS_last),
        'last_val_IBS':         float(val_IBS_last),
        'last_val_iauc':        float(val_iauc_last),
        'last_val_loss':        float(total_loss_last.item()) if hasattr(total_loss_last, 'item') else float(total_loss_last),
        'best_epoch':           int(best_epoch),
    }


def _train_val(datasets, cur, args):
    train_split, val_split = _get_splits(datasets, cur, args)
    loss_fn = _init_loss_function(args)
    model = _init_model(args)
    optimizer = _init_optim(args, model)
    train_loader, val_loader = _init_loaders(args, train_split, val_split)
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    ret = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    # 兼容两种返回格式
    if len(ret) == 3:
        results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss), last_info = ret
        try:
            import pandas as pd
            last_df = pd.DataFrame([{
                'fold':                 cur,
                'val_cindex_last':      last_info['last_val_cindex'],
                'val_cindex_ipcw_last': last_info['last_val_cindex_ipcw'],
                'val_IBS_last':         last_info['last_val_IBS'],
                'val_iauc_last':        last_info['last_val_iauc'],
                'val_loss_last':        last_info['last_val_loss'],
                'best_epoch':           last_info['best_epoch'],
            }])
            last_csv = os.path.join(args.results_dir, 'summary_last_fold{}.csv'.format(cur))
            last_df.to_csv(last_csv, index=False)
        except Exception as e:
            print(f"[warning] failed to save last-epoch summary: {e}")
    else:
        results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = ret

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)
# GRAPH_TRIAL_INSTALLED
