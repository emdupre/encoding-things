import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import base, metrics
from tqdm import tqdm


##########################################################################
# Adapted from https://github.com/leomqyu/BraInCoRL/tree/main
##########################################################################
def load_weights_and_predict(weights, x):
    """
    weights: [B, 513]
    x: [B, num_uk, 512]

    output: [B, num_uk]
    """

    weight = weights[:, :-1]
    bias = weights[:, -1]

    # print(weight.shape, bias.shape, x.shape)        # torch.Size([16, 512]) torch.Size([16]) torch.Size([16, 10, 512])
    weight = weight.unsqueeze(1)  # Shape: [16, 1, 512]
    bias = bias.unsqueeze(-1)  # Shape: [16, 1]

    pred = torch.sum(weight * x, dim=-1) + bias  # torch.Size([16, 10])

    return pred


class SwiGLUFFN(nn.Module):
    """no auto determined hidden size"""

    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear1(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.linear2(hidden)


class SwigluAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        tsfm_hidden_dim,
        num_heads,
        dropout=0.0,
        need_weights=False,
    ):
        """Conventional attention + swiglu and attention residual"""

        super().__init__()
        self.need_weights = need_weights

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, tsfm_hidden_dim, embed_dim)

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        log_scale = np.log(inp_x.shape[-2])

        """store attn weights for external access"""
        attn_output, attn_w = self.attn(
            log_scale * inp_x, inp_x, inp_x, need_weights=self.need_weights
        )
        # print('[DEBUG] attn_w.shape',attn_w.shape)      # attn_w.shape torch.Size([64, 34, 34])

        self.last_attn = attn_w
        x = x + self.attn_dropout(
            attn_output
        )  # Apply dropout to the attention output
        x = x + self.ffn(self.layer_norm_2(x))
        return x


class ResidualBlock(nn.Module):
    # Follows "Identity Mappings in Deep Residual Networks", uses LayerNorm instead of BatchNorm, and LeakyReLU instead of ReLU
    def __init__(
        self,
        feat_in=128,
        feat_out=128,
        feat_hidden=256,
        drop_out=0.0,
        use_norm=True,
    ):
        super().__init__()
        # Define the residual block with or without normalization
        if use_norm:
            self.block = nn.Sequential(
                nn.LayerNorm(feat_in),  # Layer normalization on input features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(
                    feat_in, feat_hidden
                ),  # Linear layer transforming input to hidden features
                nn.LayerNorm(
                    feat_hidden
                ),  # Layer normalization on hidden features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(
                    feat_hidden, feat_out
                ),  # Linear layer transforming hidden to output features
            )
        else:
            self.block = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(
                    feat_in, feat_hidden
                ),  # Linear layer transforming input to hidden features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(
                    feat_hidden, feat_out
                ),  # Linear layer transforming hidden to output features
            )

        # Define the bypass connection
        if feat_in != feat_out:
            self.bypass = nn.Linear(
                feat_in, feat_out
            )  # Linear layer to match dimensions if they differ
        else:
            self.bypass = (
                nn.Identity()
            )  # Identity layer if input and output dimensions are the same

    def forward(self, input_data):
        # Forward pass: apply the block and add the bypass connection
        return self.block(input_data) + self.bypass(input_data)


class HyperweightsPredictorModel(nn.Module):
    def __init__(self, backbone_type):

        super().__init__()

        backbone_configs = {
            "DINO": {
                "embed_dim": 768,
                "internal_emb_dim": 800,
                "num_tsfm_layers": 20,
                "tsfm_hidden_dim": 2048,
                "num_reg_tok": 4,
                "num_heads": 10,
                "num_early_lyr": 1,
                "num_w_pred_layers": 1,
                "early_hidden_dim": 800 * 2,
                "w_pred_hidden_dim": 800 * 2,
                "dropout": 0,
            },
            "SIGLIP": {
                "embed_dim": 1152,
                "internal_emb_dim": 1200,
                "num_tsfm_layers": 20,
                "tsfm_hidden_dim": 2048,
                "num_reg_tok": 4,
                "num_heads": 10,
                "num_early_lyr": 1,
                "num_w_pred_layers": 1,
                "early_hidden_dim": 1200 * 2,
                "w_pred_hidden_dim": 1200 * 2,
                "dropout": 0,
            },
            "CLIP": {
                "embed_dim": 512,
                "internal_emb_dim": 560,
                "num_tsfm_layers": 20,
                "tsfm_hidden_dim": 2048,
                "num_reg_tok": 4,
                "num_heads": 10,
                "num_early_lyr": 1,
                "num_w_pred_layers": 1,
                "early_hidden_dim": 560 * 2,
                "w_pred_hidden_dim": 560 * 2,
                "dropout": 0,
            },
            "vis": {
                "embed_dim": 512,
                "internal_emb_dim": 560,
                "num_tsfm_layers": 3,
                "tsfm_hidden_dim": 2048,
                "num_reg_tok": 4,
                "num_heads": 10,
                "num_early_lyr": 1,
                "num_w_pred_layers": 1,
                "early_hidden_dim": 560 * 2,
                "w_pred_hidden_dim": 560 * 2,
                "dropout": 0,
            },
        }

        if backbone_type not in backbone_configs:
            raise ValueError(
                f"Invalid backbone_type: {backbone_type}. Must be one of {list(backbone_configs.keys())}"
            )

        config = backbone_configs[backbone_type]

        self.embed_dim = config["embed_dim"]
        self.internal_emb_dim = config["internal_emb_dim"]
        self.num_tsfm_layers = config["num_tsfm_layers"]
        self.tsfm_hidden_dim = config["tsfm_hidden_dim"]
        self.num_reg_tok = config["num_reg_tok"]
        self.num_heads = config["num_heads"]
        self.num_early_lyr = config["num_early_lyr"]
        self.num_w_pred_layers = config["num_w_pred_layers"]
        self.early_hidden_dim = config["early_hidden_dim"]
        self.w_pred_hidden_dim = config["w_pred_hidden_dim"]
        self.dropout = config["dropout"]
        self.backbone_type = backbone_type

        # Print all hyperparameters during initialization
        print("\n" + "=" * 50)
        print("HyperweightsPredictorModel INITIALIZATION PARAMETERS")
        print("=" * 50)
        print(f"embed_dim: {self.embed_dim}")
        print(f"internal_emb_dim: {self.internal_emb_dim}")
        print(f"num_tsfm_layers: {self.num_tsfm_layers}")
        print(f"tsfm_hidden_dim: {self.tsfm_hidden_dim}")
        print(f"num_reg_tok: {self.num_reg_tok}")
        print(f"num_heads: {self.num_heads}")
        print(f"num_early_lyr: {self.num_early_lyr}")
        print(f"num_w_pred_layers: {self.num_w_pred_layers}")
        print(f"early_hidden_dim: {self.early_hidden_dim}")
        print(f"w_pred_hidden_dim: {self.w_pred_hidden_dim}")
        print(f"dropout: {self.dropout}")
        print("=" * 50 + "\n")

        """model struct"""
        # the first layer also used as map hidden_dim to internal hidden dim
        self.early_layers = nn.Sequential(
            ResidualBlock(
                feat_in=self.embed_dim + 1,
                feat_out=self.internal_emb_dim,
                feat_hidden=self.early_hidden_dim,
                drop_out=self.dropout,
                use_norm=True,
            ),
            *(
                ResidualBlock(
                    feat_in=self.internal_emb_dim,
                    feat_out=self.internal_emb_dim,
                    feat_hidden=self.early_hidden_dim,
                    drop_out=self.dropout,
                    use_norm=True,
                )
                for _ in range(self.num_early_lyr - 1)
            ),
        )

        # class tokens
        cls_tensor = torch.randn(1, self.num_reg_tok, self.internal_emb_dim)
        cls_tensor = cls_tensor / (float(self.internal_emb_dim + 1) ** 0.5)
        self.cls_token = nn.Parameter(cls_tensor, requires_grad=True)

        # Transformer Layers
        self.input_dropout = nn.Dropout(self.dropout)
        self.transformer = nn.Sequential(
            *(
                SwigluAttentionBlock(
                    self.internal_emb_dim,
                    self.tsfm_hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                    need_weights=True,
                )
                for _ in range(self.num_tsfm_layers)
            )
        )

        # weight prediction
        # the last layer also used as map hidden_dim to internal_hidden dim
        self.weight_pred = nn.Sequential(
            *(
                ResidualBlock(
                    feat_in=self.internal_emb_dim,
                    feat_out=self.internal_emb_dim,
                    feat_hidden=self.early_hidden_dim,
                    drop_out=self.dropout,
                    use_norm=True,
                )
                for _ in range(self.num_early_lyr - 1)
            ),
            ResidualBlock(
                feat_in=self.internal_emb_dim,
                feat_out=self.embed_dim + 1,
                feat_hidden=self.early_hidden_dim,
                drop_out=self.dropout,
                use_norm=True,
            ),
        )

    def forward(self, ic_img, ic_nrn, unknown_img):
        """
        x is a batch of image embeddings, returned is the predicted activation for the batch of image in a single voxel
        x.shape is (B, S, E) where B is batch size, S is the num of images for in context learning, E is the length of image embeddings (512)

        ic_img is the image embeddings for incontext learning: (B, S_ic, E)
        ic_nrn is the neural activation for incontext learning: (B, S_ic)
        unknown_img is the img embedding to predict: (B, S_uk, E)
        """

        # print('[DEBUG] self.dropout', self.dropout)

        B, S_ic, E = ic_img.shape  # batch, in context samples, 512

        # debug_info()
        # print(f'[DEBUG] B, S, E: {ic_img.shape}')   # B, S, E: torch.Size([1, 100, 512])

        x = self.early_layers(
            torch.cat([ic_img, ic_nrn.unsqueeze(-1)], dim=-1)
        )  # [B, S, E+1]  batch, in context samples, 512

        # # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, N, E+1]
        x = torch.cat([cls_token, x], dim=1)  # [B, S+N, E+1]

        """Apply Transformer"""
        # print('[DEBUG] type(x)', x.dtype)       # torch.float32
        x = self.input_dropout(x)
        x = self.transformer(x)

        # Perform hyperweights prediction
        pred_tok = x[:, 0, :]  # [B, E+1]

        weights = self.weight_pred(pred_tok)  # [B, E+1]

        pred = load_weights_and_predict(weights, unknown_img)

        return pred, weights


def unit_norm(a):
    if isinstance(a, np.ndarray):
        return a / np.linalg.norm(a, axis=-1, keepdims=True)
    elif isinstance(a, torch.Tensor):
        return a / torch.norm(a, p=2, dim=-1, keepdim=True)
    else:
        raise TypeError("Not numpy array or tensor")


##########################################################################


class BrainCoRL(base.BaseEstimator):
    """
    scikit-learn estimator wrapping BrainCoRL (HyperweightsPredictorModel).

    The model is a frozen in-context learner: at predict time it conditions
    on the full training set (X_fit_, Y_fit_) and returns predictions for
    new query images — no gradient update ever happens.

    Parameters
    ----------
    backbone : str
        Backbone type passed to HyperweightsPredictorModel (e.g. "CLIP").
    checkpoint_path : str
        Path to the pretrained model checkpoint (.pth).
    batch_size : int
        Number of voxels processed per forward pass.
    n_context_size : int
        Number of context samples randomly selected for each
        bootstrap iteration.

    device : str or None
        "cuda" / "cpu". If None, auto-detected.

    Attributes
    ----------
    X_fit_ : np.ndarray, shape (n_train, n_features)
    Y_fit_ : np.ndarray, shape (n_train, n_voxels)
    model_ : HyperweightsPredictorModel
    """

    def __init__(
        self,
        backbone="CLIP",
        checkpoint_path="checkpoints/CLIP_trained_on_s1257.pth",
        batch_size=512,
        nits_bootstrap=100,
        n_context_size=200,
        device=None,
    ):
        self.backbone = backbone
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.device = device
        self.nits_bootstrap = nits_bootstrap
        self.n_context_size = n_context_size
        self.model_ = self._load_model(self._get_device())

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, device):
        model = HyperweightsPredictorModel(backbone_type=self.backbone).to(
            device
        )
        checkpoint = torch.load(
            self.checkpoint_path, weights_only=True, map_location=device
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def fit(self, X, Y):
        """
        Store training data as context pool and load the frozen model.

        Parameters
        ----------
        X : array-like, shape (n, n_features)  — image embeddings
        Y : array-like, shape (n, n_voxels)    — brain responses
        """
        self.X_fit_ = np.asarray(X, dtype=np.float32)
        self.Y_fit_ = np.asarray(Y, dtype=np.float32)
        return self

    def predict(self, X):
        """
        Predict brain responses for query image embeddings.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)

        Returns
        -------
        Y_pred : np.ndarray, shape (n_query, n_voxels)
        """
        X = np.asarray(X, dtype=np.float32)
        device = self._get_device()

        X_ctx = torch.from_numpy(unit_norm(self.X_fit_)).to(device)
        Y_ctx = torch.from_numpy(self.Y_fit_).float().to(device)
        X_query = torch.from_numpy(unit_norm(X)).to(device)

        n_context = X_ctx.shape[0]
        n_voxels = Y_ctx.shape[1]
        n_query = X_query.shape[0]
        all_preds = np.zeros((n_voxels, n_query), dtype=np.float32)

        n_batches = (n_voxels + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="Processing voxels"):
                v_start = i * self.batch_size
                v_end = min(v_start + self.batch_size, n_voxels)
                bsz = v_end - v_start

                beta_ic = Y_ctx[:, v_start:v_end].T  # (bsz, n_context)

                ic_img = X_ctx.unsqueeze(0).expand(
                    bsz, -1, -1
                )  # (bsz, n_context, n_features)

                pred = torch.zeros(
                    bsz, n_query, dtype=ic_img.dtype, device=ic_img.device
                )
                for _ in range(self.nits_bootstrap):
                    indices = np.random.choice(
                        n_context, size=self.n_context_size, replace=False
                    )
                    beta_ic_bootstrap = beta_ic[
                        :, indices
                    ]  # (bsz ,n_context_size)
                    ic_img_bootstrap = ic_img[
                        :, indices, :
                    ]  # (bsz, n_context_size, n_features)
                    q_img = X_query.unsqueeze(0).expand(
                        bsz, -1, -1
                    )  # (bsz, n_query, n_features)

                    pred_bootstrap, _ = self.model_(
                        ic_img_bootstrap, beta_ic_bootstrap, q_img
                    )  # (bsz, n_query)
                    pred += pred_bootstrap

                pred /= self.nits_bootstrap
                all_preds[v_start:v_end] = pred.cpu().numpy()

        return all_preds.T  # (n_query, n_voxels)

    def score(self, X, Y):
        """R² score across all voxels."""
        Y_pred = self.predict(X)
        return metrics.r2_score(Y, Y_pred, multioutput="raw_values")
