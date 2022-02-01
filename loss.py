"""
team formation loss functions
"""
import torch


def kl_loss(mu, sigma):
    return (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()


def collaboration_score(y_pred, team_members_dict, list_loss_threshold):
    scores = []
    candidates_batch = torch.topk(y_pred, k=list_loss_threshold, sorted=True).indices
    for candidates in candidates_batch:
        y_onehot = torch.zeros(len(y_pred[0]))
        y_onehot[candidates] = 1

        cooccurance_matrix = torch.zeros((list_loss_threshold, len(y_pred[0])))
        for i, candidate in enumerate(candidates):
            try:
                cooccurance_matrix[i][team_members_dict[str(candidate.item())]] = 1
            except:
                pass
        scores.append((1/len(candidates)**2) * torch.matmul(y_onehot, torch.transpose(cooccurance_matrix, 0, 1)).sum())

    return torch.mean(torch.tensor(scores))