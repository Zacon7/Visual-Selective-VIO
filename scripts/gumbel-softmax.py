import numpy as np
import torch
from torch.autograd import Variable

# -----------------------------Softmax-Random Choice-----------------------------
def onehot_from_logits(logits, eps=0.5):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()

    # 探索率为0，则直接以概率大小选择最优操作
    if eps == 0.0:
        return argmax_acs

    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )

    # 探索率不为0，则chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )


def get_rep_outputs(logits, hard=False):
    y = torch.softmax(logits, dim=1)
    if hard:
        y_hard = onehot_from_logits(y, eps=0.5)
        y = (y_hard - y).detach() + y
    return y


# -----------------------------Gumbel-Softmax-----------------------------


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return torch.softmax(y / temperature, dim=1)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs


def get_rep_outputs(logits, temperature, hard):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


if __name__ == "__main__":
    logits = torch.tensor([0.9, 3, 2.3]).reshape(1, 3)
    print(get_rep_outputs(logits=logits, hard=True))
