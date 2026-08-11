import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    """Vector field parameterized by neural dynamics for Adjoint backpropagation."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, state):
        return self.net(state)

class SPINN_Engine(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, pred_len=6):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.lif = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.ode_func = ODEFunc(hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        mem = self.lif.init_leaky()
        mem_history = []
        
        for t_step in range(x.size(1)):
            spk, mem = self.lif(self.encoder(x[:, t_step, :]), mem)
            mem_history.append(mem)
            
        h = mem_history[-1] 
        t_span = torch.linspace(0, 1, steps=self.pred_len).to(x.device)
        ode_out = odeint(self.ode_func, h, t_span, method='euler')
        return self.head(ode_out.permute(1, 0, 2))

class GRU_Baseline(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, pred_len=6):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len * input_dim)
        
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0)).view(x.size(0), self.pred_len, self.input_dim)

def physics_loss_kernel(pred_tensor, device):
    w_wnd, w_tmp, w_dew = torch.tensor([-0.05, 0.02, 0.01], dtype=torch.float32, device=device)
    slp_pred = pred_tensor[:, :, 0]
    slp_derived = (pred_tensor[:, :, 1] * w_wnd) + (pred_tensor[:, :, 2] * w_tmp) + (pred_tensor[:, :, 3] * w_dew)
    return torch.mean((slp_pred - slp_derived) ** 2)
