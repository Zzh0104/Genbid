from pathlib import Path
import numpy as np
import pytest
import torch
from ogb import Episode, Policy, Replay, DT, FEATURES, SCHEMA
from strategy import OgbBiddingStrategy

def test_adapter_matches_replay_state_and_rtg(monkeypatch, tmp_path):
    checkpoint = tmp_path/'model.pt'
    model = DT()
    torch.save(dict(model=model.state_dict(), config=model.config, schema=SCHEMA,
                    features=FEATURES, mean=[0.]*len(FEATURES), std=[1.]*len(FEATURES),
                    return_scale=1., target=5., context=8, penalty=2.), checkpoint)
    monkeypatch.setenv('OGB_CHECKPOINT', str(checkpoint))
    agent = OgbBiddingStrategy(budget=70, cpa=15)
    direct = Policy(checkpoint)
    p = np.array([.01,.03,.07])
    prices = np.array([.1,.3,.7])
    env = Replay(Episode('test',70,15,[(p,prices)]*48), penalty=2.)
    hp,hb,ha,hi,hl = [],[],[],[],[]
    prev = 0.
    for t in range(3):
        expected = direct.act(env.state(),t,prev)
        bid = agent.bidding(t,p,np.zeros(3),hp,hb,ha,hi,hl)
        np.testing.assert_allclose(bid,expected*15*p,rtol=1e-5)
        won = bid >= prices
        prev = env.step(expected)
        agent.remaining_budget = 70-env.cost
        hp.append(np.column_stack((p,np.zeros(3))))
        hb.append(bid)
        ha.append(np.column_stack((won,won,prices*won)))
        hi.append(np.zeros((3,2)))
        hl.append(prices)
    agent.reset()
    assert agent.remaining_budget == 70
    assert not agent.policy.states
