import torch
import torch.nn as nn


class LearnedPositionEmbedding(nn.Module):

    def __init__(self, seq_len, embed_dim) -> None:
        """
        Adds position embedding to the input vectors.
        """
        super(LearnedPositionEmbedding, self).__init__()
        self.pos_embed = nn.parameter.Parameter(torch.Tensor(seq_len, embed_dim))
        torch.nn.init.kaiming_normal_(self.pos_embed)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 3, "Number of dimensions should be 3 [batch, seq_len, f_len]"

        # pos embedding shoud be broadcastable.
        out = input + self.pos_embed  # (batch, seq_len, f_len)
        return out


# Attention-Fusion for the visual and imu features:
class EarlyFusionAttention(nn.Module):
    """
    Refer to: 'Wayformer: Motion Forecasting via Simple & Efficient Attention Networks'
    https://arxiv.org/pdf/2207.05844.pdf
    For more general architecture refer to DeepMind's 'Perceiver: General Perception with Iterative Attention'.
    https://arxiv.org/abs/2103.03206
    """

    def __init__(self, opt, num_blocks=1, num_heads=2, pos_embed_req=True):
        """
        This is more like cross-attention.
        Cross-attends features from different modalities.

        Args:
            - opt: config args required for training.
            - num_blocks: number of attention blocks
            - num_heads: number of attention head per block
        """
        super(EarlyFusionAttention, self).__init__()
        embed_dims = opt.v_f_len + opt.i_f_len
        self.num_blocks = num_blocks

        self.attn_blocks = []
        for _ in range(self.num_blocks):
            self.attn_blocks.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dims,
                    num_heads=num_heads,
                    batch_first=True,
                )
            )

        for attn_block in self.attn_blocks:
            attn_block = attn_block.to("cuda:0")

        self.pos_embed = None
        if pos_embed_req:
            self.pos_embed = LearnedPositionEmbedding(
                seq_len=1,
                embed_dim=embed_dims
            )

    def forward(self, v_features, i_features):
        """
        Concatenate the features and apply attention.
        """
        concat_vi = torch.cat((v_features, i_features), dim=-1)   # (batch, seq_len, v_len+i_len)

        x = self.pos_embed(concat_vi) if self.pos_embed else concat_vi  # (batch, seq_len, v_len+i_len)

        for attn_block in self.attn_blocks:
            x, _ = attn_block(query=x, key=x, value=x)

        return x + concat_vi  # (batch, seq_len, v_len+i_len)
