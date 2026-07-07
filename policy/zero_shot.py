import torch
import math
import argparse
from model.model import Model


class ZeroShotPolicy:
    """Extracts encoder + scorer from pretrained model for per-step action selection.

    The original Decoder.forward() owns its Env loop internally. This class
    separates out encoder and scoring weights so an external environment
    (MCEnv) can drive the loop, calling get_action() at each step.
    """

    def __init__(self, model_path='baselines/models/proposed/epoch(100).pt',
                 device=torch.device('cpu')):
        self.device = device

        args = argparse.Namespace(
            device=device, embed_dim=128, n_encode_layers=3, n_heads=8,
            ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
            online=False, online_known_num=None
        )
        full_model = Model(args)
        state = torch.load(model_path, map_location=device)
        full_model.load_state_dict(state)
        full_model.eval()

        self.encoder = full_model.decoder.encoder
        self.encoder.eval()

        self.tanh_c = full_model.decoder.tanh_c
        self.W_target = full_model.decoder.W_target
        self.W_global = full_model.decoder.W_global
        self.W_K1 = full_model.decoder.W_K1
        self.W_K2 = full_model.decoder.W_K2
        self.W_Q = full_model.decoder.W_Q
        self.W_V = full_model.decoder.W_V
        self.MHA = full_model.decoder.MHA

        for m in [self.W_target, self.W_global, self.W_K1, self.W_K2,
                  self.W_Q, self.W_V, self.MHA]:
            m.eval()

    @torch.no_grad()
    def get_scores(self, x, n_bays, n_rows, n_tiers, target_stack, invalid_mask=None,
                   t_acc=40, t_bay=3.5, t_row=1.2, t_pd=30):
        batch = 1
        max_stacks = n_bays * n_rows

        node_embeddings, graph_embedding = self.encoder(
            x.reshape(batch, max_stacks, n_tiers), n_bays, n_rows,
            t_acc=t_acc, t_bay=t_bay, t_row=t_row, t_pd=t_pd
        )

        target_emb = node_embeddings[:, target_stack:target_stack+1, :]
        context = (self.W_target(target_emb) + self.W_global(graph_embedding.unsqueeze(1)))

        node_keys = self.W_K1(node_embeddings)
        node_values = self.W_V(node_embeddings)
        query = self.W_Q(self.MHA([context, node_keys, node_values]))
        key = self.W_K2(node_embeddings)

        logits = torch.matmul(query, key.permute(0, 2, 1)).squeeze(1) / math.sqrt(query.size(-1))
        logits = self.tanh_c * torch.tanh(logits)
        if invalid_mask is not None:
            logits = logits - invalid_mask.float() * 1e9
        return torch.log_softmax(logits, dim=1)

    @torch.no_grad()
    def get_action(self, x, n_bays, n_rows, n_tiers, target_stack, invalid_mask=None,
                   t_acc=40, t_bay=3.5, t_row=1.2, t_pd=30):
        log_p = self.get_scores(x, n_bays, n_rows, n_tiers, target_stack, invalid_mask,
                                t_acc=t_acc, t_bay=t_bay, t_row=t_row, t_pd=t_pd)
        return log_p.argmax(dim=1, keepdim=True)
