import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualInformationLoss:
    def __init__(self, bins=32, range=None):
        self.bins = bins
        self.range = range
        self.tensor1 = None
        self.tensor2 = None

    def set_tensors(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def _calculate_histograms(self):
        # Flattening tensors
        t1_flat = self.tensor1.flatten()
        t2_flat = self.tensor2.flatten()

        # Calculating joint histogram
        joint_hist = torch.histc(t1_flat.unsqueeze(1) * self.bins + t2_flat, bins=self.bins**2, min=0, max=self.bins**2)
        joint_hist = joint_hist.reshape(self.bins, self.bins)  # Reshape to bins x bins
        
        # Marginal histograms
        hist1 = torch.sum(joint_hist, dim=1)
        hist2 = torch.sum(joint_hist, dim=0)

        return joint_hist, hist1, hist2

    def _calculate_mutual_information(self, joint_hist, hist1, hist2):
        # Convert histograms to probabilities
        joint_prob = joint_hist / torch.sum(joint_hist)
        prob1 = hist1 / torch.sum(hist1)
        prob2 = hist2 / torch.sum(hist2)

        # Calculate entropies
        def entropy(prob):
            return -torch.sum(prob[prob > 0] * torch.log2(prob[prob > 0]))

        H1 = entropy(prob1)
        H2 = entropy(prob2)
        H12 = entropy(joint_prob.flatten())

        # Mutual information
        MI = H1 + H2 - H12
        return MI

    def get_mutual_information_loss(self):
        joint_hist, hist1, hist2 = self._calculate_histograms()
        mi = self._calculate_mutual_information(joint_hist, hist1, hist2)
        return (1-mi)/2

class ODSLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(ODSLoss, self).__init__()
        self.device = device

    def forward(self, z1, z2):
        # Move tensors to the specified device
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)

        # Reshape and perform SVD as before
        z1_reshaped = z1.view(z1.size(1), -1)
        z2_reshaped = z2.view(z2.size(1), -1)

        U1, S1, V1 = torch.svd(z1_reshaped)
        U2, S2, V2 = torch.svd(z2_reshaped)

        loss_U = torch.norm(torch.matmul(U1.transpose(-2, -1), U2), p='fro')
        S1_matrix = torch.diag_embed(S1)
        S2_matrix = torch.diag_embed(S2)
        loss_S = torch.norm(torch.matmul(S1_matrix.transpose(-2, -1), S2_matrix), p='fro')
        loss_V = torch.norm(torch.matmul(V1.transpose(-2, -1), V2), p='fro')

        total_loss = loss_U + loss_S + loss_V

        return total_loss

def ods_loss(z1, z2):
    # Reshape and perform SVD as before
    z1_reshaped = z1.view(z1.size(1), -1)
    z2_reshaped = z2.view(z2.size(1), -1)

    U1, S1, V1 = torch.svd(z1_reshaped)
    U2, S2, V2 = torch.svd(z2_reshaped)

    loss_U = torch.norm(torch.matmul(U1.transpose(-2, -1), U2), p='fro')
    S1_matrix = torch.diag_embed(S1)
    S2_matrix = torch.diag_embed(S2)
    loss_S = torch.norm(torch.matmul(S1_matrix.transpose(-2, -1), S2_matrix), p='fro')
    loss_V = torch.norm(torch.matmul(V1.transpose(-2, -1), V2), p='fro')

    total_loss = loss_U + loss_S + loss_V

    return total_loss

class CCALoss(nn.Module):
    def __init__(self, device='cuda', reg_lambda=1e-5):
        super(CCALoss, self).__init__()
        self.device = device
        self.reg_lambda = reg_lambda
        input_dim = 4 * 32 * 32  # Adjust based on the input shape
        output_dim = 128  # You can tune this parameter

        self.linear_z0 = nn.Linear(input_dim, output_dim).to(device)
        self.linear_z1 = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, z0, z1):
        # Flatten the inputs
        z0_flat = z0.view(z0.size(0), -1).to(self.device)
        z1_flat = z1.view(z1.size(0), -1).to(self.device)

        # Apply the linear transformations
        z0_transformed = self.linear_z0(z0_flat)
        z1_transformed = self.linear_z1(z1_flat)

        # Normalize the transformed outputs
        z0_normalized = z0_transformed - z0_transformed.mean(dim=0)
        z1_normalized = z1_transformed - z1_transformed.mean(dim=0)

        # Compute covariance matrices
        cov_z0 = torch.matmul(z0_normalized.T, z0_normalized)
        cov_z1 = torch.matmul(z1_normalized.T, z1_normalized)
        cov_z0z1 = torch.matmul(z0_normalized.T, z1_normalized)
        # Add regularization to the diagonal of the covariance matrices
        reg = torch.eye(cov_z0.size(0)).to(self.device) * self.reg_lambda
        cov_z0 += reg
        cov_z1 += reg

        # Compute the CCA loss (minimizing correlation)
        cca_loss = -torch.trace(torch.matmul(torch.linalg.inv(cov_z0), cov_z0z1)
                                @ torch.matmul(torch.linalg.inv(cov_z1), cov_z0z1.T))

        return cca_loss
    
    # import torch
from torch import nn


class RBF(nn.Module):

    def __init__(self, n_kernels=4, mul_factor=2.0, bandwidth=None, device="cuda"):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels).to(self.device) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, device = "cuda"):
        super().__init__()
        self.kernel = RBF(device = device)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    
class MMDloss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMDloss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None   

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0, "Batch size must be non-zero."

        # Ensure mu and logvar are 2D (batch_size, num_features)
        if mu.data.ndimension() != 2:
            raise ValueError("mu should be of shape (batch_size, num_features)")
        if logvar.data.ndimension() != 2:
            raise ValueError("logvar should be of shape (batch_size, num_features)")

        # klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # total_kld = klds.sum(1).mean(0, keepdim=True)
        # dimension_wise_kld = klds.mean(0)
        # mean_kld = klds.mean(1).mean(0, keepdim=True)
        
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kld.mean()
    

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
    
    def forward(self, x, y):
        # Normalize x and y to unit vectors along the last dimension
        x_normalized = F.normalize(x, p=2, dim=1)
        y_normalized = F.normalize(y, p=2, dim=1)
        
        # Compute cosine similarity along the last dimension
        cosine_similarity = torch.sum(x_normalized * y_normalized, dim=1)
        
        # Since we want to maximize the cosine similarity, and PyTorch minimizes the loss,
        # we can subtract the cosine similarity from 1 to convert it into a loss.
        # We take the mean loss over all samples in the batch.
        return 1 - cosine_similarity.mean()

# Example usage:
if __name__ == "__main__":
    # Creating dummy tensors
    tensor1 = torch.randn(3, 512)
    tensor2 = torch.randn(3, 512)

    # Instantiating the loss class
    loss_fn = CosineSimilarityLoss()

    # Computing the loss
    loss = loss_fn(tensor1, tensor2)
    print("Cosine Similarity Loss:", loss.item())