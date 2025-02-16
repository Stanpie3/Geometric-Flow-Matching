import torch

def Euler(ut, x_init, t_start=0., t_end=1., num_steps=100, device='cpu'):
    xt = x_init
    t = t_start*torch.ones((xt.shape[0],1)).to(device)
    dt = (t_end-t_start)/num_steps

    with torch.no_grad():
      for i in range(num_steps):
        xt = xt + ut(xt,t) * dt
        t = t + dt

    return xt

def RK4(ut, x_init, t_start=0., t_end=1., num_steps=100, device='cpu'):
    xt = x_init
    device = x_init.device
    t = t_start * torch.ones((xt.shape[0], 1), device=device)
    dt = (t_end - t_start) / num_steps

    with torch.no_grad():
        for _ in range(num_steps):
            k1 = ut(xt, t)
            k2 = ut(xt + dt/2.0 * k1, t + dt/2.0)
            k3 = ut(xt + dt/2.0 * k2, t + dt/2.0)
            k4 = ut(xt + dt * k3, t + dt)
            xt = xt + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
            t = t + dt

    return xt