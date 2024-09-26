import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F


class PITNet1(nn.Module):
    def __init__(self):
        super(PITNet1, self).__init__()
        self.lamda1 = nn.Parameter(torch.tensor([1e1]), requires_grad=True)
        self.rho = nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        self.w = nn.Parameter(torch.randn(471))
        self.b1 = nn.Parameter(torch.randn(471))
        self.linear = nn.Linear(471, 471)

        init.constant_(self.lamda1, 1e1)
        init.constant_(self.rho, 0.01)
        init.normal_(self.w, mean=0.0015, std=0.0001)

        self.First_layer = First_Layer(self.lamda1, self.w)
        self.theta_update_layer = theta_Update_Layer(self.lamda1, self.w)
        self.z_update_layer = z_Update_Layer(self.rho, self.b1, self.w, self.linear)
        self.u_update_layer = u_Update_Layer(self.w)
        self.final_layer = final_Layer(self.rho, self.b1, self.w, self.linear)
        self.layers = nn.ModuleList()

        self.layers.append(self.First_layer)
        self.layers.append(self.z_update_layer)
        self.layers.append(self.u_update_layer)
        for i in range(4):
            self.layers.append(self.theta_update_layer)
            self.layers.append(self.z_update_layer)
            self.layers.append(self.u_update_layer)
        self.layers.append(self.theta_update_layer)
        self.layers.append(self.final_layer)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x, q_t):
        out = (x, q_t)
        for layer in self.layers:
            out = layer(*out)
        # Compute the sum of all elements in b
        b_sum = torch.sum(out)
        # Count the number of non-zero elements in b
        non_zero_count = torch.nonzero(out).size(0)
        return out


class First_Layer(nn.Module):
    def __init__(self, lamda1, w):
        super(First_Layer, self).__init__()
        self.lamda1 = lamda1
        self.w = w

    def forward(self, x, q_t):

        z = torch.zeros(471)
        u = torch.zeros(471)
        v = z - self.w + u
        theta = self.lamda1 * torch.matmul(torch.pinverse(torch.matmul(q_t, q_t.t())), torch.matmul(q_t, v))
        PIT_data = dict()
        ATA = torch.matmul(x.T, x)
        PIT_data['A_Transfer'] = ATA
        PIT_data['input'] = x
        PIT_data['theta'] = theta
        PIT_data['u'] = torch.zeros(471)

        return PIT_data, q_t


class theta_Update_Layer(nn.Module):
    def __init__(self, lamda1, w):
        super(theta_Update_Layer, self).__init__()
        self.lamda1 = lamda1
        self.w = w

    def forward(self, PIT_data, q_t):
        z = PIT_data['z']
        u = PIT_data['u']
        v = z - self.w + u
        term1 = torch.matmul(q_t, q_t.t())
        theta = self.lamda1 * torch.matmul(torch.pinverse(term1), torch.matmul(q_t, v))
        PIT_data['theta'] = theta

        return PIT_data, q_t


class z_Update_Layer(nn.Module):
    def __init__(self, rho, b1, w, linear, topk=50):
        super(z_Update_Layer, self).__init__()
        self.rho = rho
        self.linear = linear
        self.topk = topk
        self.b1 = b1
        self.w = w

    def forward(self, PIT_data, q_t):

        theta = PIT_data['theta']
        u = PIT_data['u']
        A = PIT_data['input']
        N = 471
        ATA = PIT_data['A_Transfer']
        W1 = self.linear(ATA)

        term1 = self.linear(A)
        W2 = term1.mean(dim=0)

        v = W2 + self.rho * (self.w + (1 / N) * torch.matmul(theta, q_t) - u)
        z = torch.matmul(W1, v)
        z = F.relu(z)
        # 挑选前k个元素
        _, indices = torch.topk(z, self.topk, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=z.dtype))
        z = z * mask

        epsilon = 1e-8     # 归一化处理
        z = z / (torch.sum(z) + epsilon)
        PIT_data['z'] = z
        return PIT_data, q_t


class u_Update_Layer(nn.Module):
    def __init__(self, w):
        super(u_Update_Layer, self).__init__()
        self.w = w

    def forward(self, PIT_data, q_t):
        u = PIT_data['u']
        theta = PIT_data['theta']
        N = 471
        z = PIT_data['z']
        output = u + z - (self.w + (1 / N) * torch.matmul(theta, q_t))
        PIT_data['u'] = output
        return PIT_data, q_t


class final_Layer(nn.Module):
    def __init__(self, rho, b1, w, linear, topk=50):
        super(final_Layer, self).__init__()
        self.rho = rho
        self.topk = topk
        self.b1 = b1
        self.linear = linear
        self.w = w

    def forward(self, PIT_data, q_t):
        theta = PIT_data['theta']
        u = PIT_data['u']
        ATA = PIT_data['A_Transfer']
        N = 471
        I = torch.eye(N)

        temp = torch.inverse(ATA + self.rho * I)
        v = self.b1 + self.rho * (self.w + (1 / N) * torch.matmul(theta, q_t) - u)
        z = torch.matmul(temp, v)

        z = F.relu(z)
        # 挑选前k个元素
        _, indices = torch.topk(z, self.topk, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=z.dtype))
        z = z * mask

        epsilon = 1e-8  # 归一化处理
        z = z / (torch.sum(z) + epsilon)
        PIT_data['z'] = z
        return z



