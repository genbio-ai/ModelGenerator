from modelgenerator.huggingface_models.rnabert import RNABertForMaskedLM
import torch
import gc
import numpy as np
from matplotlib import pyplot as plt

import os
import torch
import torch.nn.functional as F
import sys

vocab_a2n_lm = {
    'A': 5,
    'G': 6,
    'C': 7,
    'U': 9,
    '[MASK]': 1,
    '[PAD]': 0, 
    # 'N': 10, ## for inverse_folding we are ignoring non-standard nucleotides.
    # '[UNK]': 4, ## there should be no unknown tokens. it must have been addressed in the preprocessing steps. 
}
vocab_n2a_lm = {n:a for a,n in vocab_a2n_lm.items()}

# List of possible RNA nucleotides
vocab_a2n_grnade = {
    'A': 0, 
    'G': 1, 
    'C': 2, 
    'U': 3,
    '[MASK]': 4,
}
vocab_n2a_grnade = {n:a for a,n in vocab_a2n_grnade.items()}

vocab_grnade2lm = {
    0:5,    #A
    1:6,    #G
    2:7,    #C
    3:9,    #U
    4:1,    #[MASK]
}
vocab_lm2grnade = {n:a for a,n in vocab_grnade2lm.items()}

def compute_entropy(probs=None, logits=None):
    if probs is None:
        probs = F.softmax(logits, dim=-1)  # B x N x D
    log_probs = torch.log(probs)  # B x N x D
    entropy = -torch.sum(probs * log_probs, dim=-1) # B x N
    return entropy

def get_masked_seq(seq, logits, M, mask_token=vocab_a2n_lm['[MASK]']):
    seq = seq.detach()
    logits = logits.detach()
    probab = F.softmax(logits, dim=-1).max(-1).values
    
    mask = torch.zeros_like(seq)  # B x N
    for i in range(seq.shape[0]):
        topk_indices = probab[i].argsort(-1)[:M[i]]
        mask[i].scatter_(0, topk_indices, 1)
    
    seq_masked = (mask * mask_token + (1 - mask) * seq).to(seq)

    cls_bos = torch.tensor([[2, 11]])  # B x 2
    eos_sep = torch.tensor([[12, 3]])  # B x 2

    seq_masked = torch.cat([cls_bos, seq_masked, eos_sep], -1)   # B x (2+N+2)
    seq_masked = seq_masked.squeeze(1)
    mask = mask.squeeze(1)
    return seq_masked, mask

## load the model with the maskedlm head
pretrained_LM_location = "genbio-ai/AIDO.RNA-1.6B"
device = 'cuda'
model = RNABertForMaskedLM.from_pretrained(pretrained_LM_location).to(device)
model = model.eval()

#####
split = 'test'
MGEN_DATA_DIR = os.getenv("MGEN_DATA_DIR")
DATA_ROOT = MGEN_DATA_DIR+'/modelgenerator/datasets/rna_inv_fold/structure_encoding/'
logits_all, estimated_seq_by_grnade_all = torch.load(DATA_ROOT + f'logits_samples__{split}.pt', weights_only=False)
lm_labels_all, encodings_all = torch.load(DATA_ROOT + f'encodings__{split}.pt', weights_only=False) ## load labels
logits_all = [torch.from_numpy(l).float() for l in logits_all] ## initial logits
lm_labels_all = [torch.Tensor([vocab_grnade2lm[t] for t in l]).long() for l in lm_labels_all]
attention_mask_all = [torch.ones_like(l).float() for l in lm_labels_all]

assert lm_labels_all[0].shape[0] == logits_all[0].shape[1] 
assert len(lm_labels_all[0].shape) == 1
assert len(logits_all[0].shape) == 3
assert logits_all[0].shape[-1] == 4

### delete what we dont need right now
del estimated_seq_by_grnade_all
del encodings_all

## select whether to run the full ablation or not
try:
    ablation = sys.argv[1]
except:
    ablation = ''

## NOTE: You can tune the following two hyperparameters. We also provide the best ones we found form our ablation.
if ablation == 'run_ablation':
    ## to run the full ablation
    num_denoise_steps = [1, 5, 10, 30, 50]
    MASK_RATIOs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
else:
    ## best hyperparameters
    num_denoise_steps = [5]
    MASK_RATIOs = [0.7]
    
file_writable_string = 'num_denoise_step\tMASK_RATIO\tZeroshot_Mean\tZeroshot_Median\tBaseline_Mean\tBaseline_Median\n'

for MASK_RATIO in MASK_RATIOs:
    for num_denoise_step in num_denoise_steps:
        print(f">> num_denoise_step: {num_denoise_step},\tMASK_RATIO: {MASK_RATIO}")
        
        accuracies = torch.ones((len(lm_labels_all), num_denoise_step+1)) * (-1)
        accuracies_list = torch.ones((len(lm_labels_all),2)) * (-1)
        
        baseline_acc = []
        log_mixing_acc = []
        
        for i in range(len(lm_labels_all)):
            lm_labels = lm_labels_all[i].unsqueeze(0)
            attention_mask = attention_mask_all[i].unsqueeze(0)
            logitsS0 = logits = logits_all[i].sum(0).unsqueeze(0)
            probab = F.softmax(logits, dim=-1).max(-1).values

            ## eval S0 (gRNAde's prediction)
            seq_pred = logits.argmax(-1)
            seq_pred = torch.Tensor([vocab_grnade2lm[t.item()] for t in seq_pred[0]]).unsqueeze(0).long()
            s0_acc = acc = (seq_pred == lm_labels).float().mean().cpu().item()
            baseline_acc += [acc]
            
            ## compute the number of masked tokens for the current sequence (always integer; minimum is 1)
            M = [max(1, int(attention_mask.sum(-1) * MASK_RATIO))] # shape = (B,) 
            
            ## denoising steps
            for denoise_step in range(1, num_denoise_step+1):
                masked_seq, mask = get_masked_seq(seq=seq_pred, logits=logits, M=M, mask_token=vocab_a2n_lm['[MASK]'])
                # print(seq_pred, masked_seq)
                output = model(input_ids=masked_seq.to(device), attention_mask=torch.ones_like(masked_seq).to(device))
                # logits = output.logits[:, 2:-2, [5,6,7,9]].detach().cpu() ## taking only the canonical nucleotides; directly maps back to gRNAde's vocab
                logits = output.logits[:, 2:-2, ].detach().cpu() ## taking only the canonical nucleotides; directly maps back to gRNAde's vocab
                seq_pred = logits.argmax(-1)
                # print(seq_pred)
            logits = logits[:, :, [5,6,7,9]].detach().cpu() ## taking only the canonical nucleotides; directly maps back to gRNAde's vocab
            lm_labels = torch.tensor([vocab_lm2grnade[t.item()] for t in lm_labels[0]])
            logits = logits + logitsS0 ## NOTE: additive bias, inspired by LM-Design paper (ref: https://doi.org/10.48550/arXiv.2302.01649); ## if we directly use logits, while it can maximize the pseudo-log-likelihood (i.e., giving us a more likely natural RNA), it may deviate from the sequence we actually want from inverse folding since there is no constraint.
            seq_pred = logits.argmax(-1)
            acc = (seq_pred == lm_labels).float().mean().cpu().item()
            log_mixing_acc += [acc]
                
        lengths = torch.Tensor([l.sum() for l in attention_mask_all])
        print(f'\tZeroshot Mean: {np.mean(log_mixing_acc)}\tZeroshot Median: {np.median(log_mixing_acc)}\n\tBaseline Mean: {np.mean(baseline_acc)}\tBaseline Median: {np.median(baseline_acc)}')
        file_writable_string += f'{num_denoise_step}\t{MASK_RATIO}\t{np.mean(log_mixing_acc)}\t{np.median(log_mixing_acc)}\t{np.mean(baseline_acc)}\t{np.median(baseline_acc)}\n'

OUTDIR = "rnaIF_outputs"
os.makedirs(OUTDIR, exist_ok=True)
with open(OUTDIR+'/zeroshot_analyses_results.tsv', 'w') as f:
    f.write(file_writable_string)

print('Done!')
