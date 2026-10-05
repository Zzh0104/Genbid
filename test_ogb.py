import numpy as np
import torch
import pytest
from ogb import DT, Episode, FEATURES, Replay, mpc, trajectory

def test_no_action_or_future_leakage():
    torch.manual_seed(3)
    model = DT().eval()
    s, a, r, t = torch.randn(1, 5, len(FEATURES)), torch.rand(1, 5, 1), torch.rand(1, 5, 1), torch.arange(5)[None]
    with torch.no_grad():
        pred = model(s, a, r, t)
        a[:, 2:] = 100
        s[:, 3:] = 100
        r[:, 3:] = 100
        changed = model(s, a, r, t)
    torch.testing.assert_close(pred[:, :3], changed[:, :3])

def test_budget_and_rtg():
    e = Episode('0:0', 1., 10., [(np.array([.2,.2,.2]), np.array([.7,.7,.7]))]*3)
    env = Replay(e)
    for _ in e.slots:
        env.step(2.)
    assert env.cost <= e.budget
    tr = trajectory(e, 'mpc')
    np.testing.assert_allclose(tr['rtg'][:,0], np.cumsum(tr['rewards'][::-1])[::-1], atol=1e-6)

def test_state_cannot_see_current_prices_or_future():
    a = Episode('0:0', 10., 10., [(np.array([.1]), np.array([1.]))]*3)
    b = Episode('1:0', 10., 10., [(np.array([.1]), np.array([999.]))]*3)
    np.testing.assert_array_equal(Replay(a).state(), Replay(b).state())

def test_padding_does_not_change_valid_outputs():
    model = DT().eval()
    s, a, r = torch.rand(1,4,len(FEATURES)), torch.rand(1,4,1), torch.rand(1,4,1)
    t = torch.arange(4)[None]
    with torch.no_grad():
        full = model(s,a,r,t,torch.tensor([[True,True,False,False]]))
        short = model(s[:,:2],a[:,:2],r[:,:2],t[:,:2])
    torch.testing.assert_close(full[:,:2], short)

def test_mpc_returns_grid_action():
    env = Replay(Episode('0:0', 10., 10., [(np.array([.1]),np.array([.5]))]*3))
    assert 0 <= mpc(env) <= 2

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_cuda_inference_agree():
    model = DT().eval()
    s,a,r,t = torch.rand(1,5,len(FEATURES)),torch.rand(1,5,1),torch.rand(1,5,1),torch.arange(5)[None]
    with torch.no_grad():
        cpu = model(s,a,r,t)
        gpu = model.cuda()(s.cuda(),a.cuda(),r.cuda(),t.cuda()).cpu()
    torch.testing.assert_close(cpu,gpu,atol=1e-5,rtol=1e-5)
