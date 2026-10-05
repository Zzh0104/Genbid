"""Public OGB reconstruction: replay MPC teacher -> causal DT -> RTG serving.

Independent implementation of the DT token convention, no upstream code copied.
"""
from __future__ import annotations
import argparse
import itertools
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = ['time_left', 'budget_left', 'pacing_gap', 'last_multiplier',
            'cpa_ratio', 'conversion_per_budget', 'cost_per_budget',
            'current_pvalue_mean', 'current_volume', 'past_price_over_cpa']
SCHEMA = 'ogb-public-v1'
GRID = np.array([0., .5, .75, 1., 1.25, 1.5, 2.])

@dataclass
class Episode:
    key: str
    budget: float
    cpa: float
    slots: list

def synthetic(seed=42, periods=24, steps=16):
    rng = np.random.default_rng(seed)
    episodes = []
    for day in range(periods):
        for adv in range(3):
            cpa = float(rng.uniform(10, 25))
            slots = []
            for t in range(steps):
                p = rng.uniform(.005, .09, 64)
                price = cpa * p * rng.lognormal(-.2 + .35*np.sin(t/3), .4, len(p))
                slots.append((p, price))
            episodes.append(Episode(f'{day}:{adv}', 70., cpa, slots))
    return episodes

def read_csv(paths):
    """Whole periods are required; no arbitrary row truncation of episodes."""
    episodes = []
    required = ['deliveryPeriodIndex', 'advertiserNumber', 'budget', 'CPAConstraint',
                'timeStepIndex', 'pvIndex', 'pValue', 'leastWinningCost']
    seen = set()
    for path in paths:
        df = pd.read_csv(path, usecols=required)
        if df[required].isna().any().any():
            raise ValueError('CSV contains missing fields')
        if not np.isfinite(df[required].to_numpy(dtype=float)).all():
            raise ValueError('CSV contains nonfinite fields')
        for (day, adv), g in df.groupby(required[:2], sort=True):
            key = f'{int(day)}:{int(adv)}'
            if key in seen:
                raise ValueError(f'Duplicate episode {key}; put each full period in one file')
            seen.add(key)
            if g.budget.nunique() != 1 or g.CPAConstraint.nunique() != 1:
                raise ValueError('Budget/CPA must be constant within an episode')
            slots = []
            times = sorted(g.timeStepIndex.unique())
            if times != list(range(48)):
                raise ValueError(f'{key}: requires all 48 steps; partial logged episodes are unsupported')
            for _, tick in g.groupby('timeStepIndex', sort=True):
                tick = tick.sort_values('pvIndex')
                if tick.pvIndex.duplicated().any():
                    raise ValueError('Duplicate opportunity')
                p, price = tick.pValue.to_numpy(), tick.leastWinningCost.to_numpy()
                if ((p < 0) | (p > 1)).any() or (price < 0).any():
                    raise ValueError('Invalid pValue/price')
                slots.append((p, price))
            budget, cpa = float(g.budget.iloc[0]), float(g.CPAConstraint.iloc[0])
            if budget <= 0 or cpa <= 0:
                raise ValueError('Budget and CPA must be positive')
            episodes.append(Episode(key, budget, cpa, slots))
    return episodes

class Replay:
    """Fixed-price expected-conversion replay, chronological hard budget cap."""
    def __init__(self, episode, penalty=1.):
        self.e, self.penalty = episode, penalty
        self.t, self.cost, self.conv, self.last = 0, 0., 0., 1.
        self._outcome_cache = {}

    def state(self):
        p, _ = self.e.slots[self.t]
        past = self.e.slots[:self.t]
        price = np.mean(np.concatenate([x[1] for x in past])) if past else 0.
        return np.array([1-self.t/len(self.e.slots), 1-self.cost/self.e.budget,
                         self.cost/self.e.budget-self.t/len(self.e.slots), self.last,
                         self.cost/max(self.conv*self.e.cpa, 1e-6),
                         self.conv*self.e.cpa/self.e.budget, self.cost/self.e.budget,
                         p.mean() if len(p) else 0., np.log1p(len(p)), price/self.e.cpa], dtype=np.float32)

    def outcome(self, t, action, remaining):
        key = (t, float(action))
        if key not in self._outcome_cache:
            p, price = self.e.slots[t]
            eligible = action*self.e.cpa*p >= price
            self._outcome_cache[key] = (p[eligible], price[eligible])
        p, price = self._outcome_cache[key]
        if price.sum() <= remaining:
            return float(price.sum()), float(p.sum())
        cost, conv = 0., 0.
        for prob, payment in zip(p, price):
            if payment <= remaining-cost + 1e-10:
                cost += float(payment)
                conv += float(prob)
        return cost, conv

    def utility(self, cost, conv):
        return conv - self.penalty*max(cost/self.e.cpa-conv, 0.)

    def step(self, action):
        old = self.utility(self.cost, self.conv)
        cost, conv = self.outcome(self.t, float(np.clip(action, 0, 2)), self.e.budget-self.cost)
        self.cost += cost
        self.conv += conv
        self.last = float(np.clip(action, 0, 2))
        self.t += 1
        return self.utility(self.cost, self.conv)-old

def mpc(env, horizon=2):
    """Hindsight teacher ONLY: future prices/pvalues never enter DT state."""
    h = min(horizon, len(env.e.slots)-env.t)
    best, best_action = -float('inf'), 0.
    for actions in itertools.product(GRID, repeat=h):
        cost, conv = env.cost, env.conv
        for offset, action in enumerate(actions):
            dc, dv = env.outcome(env.t+offset, action, env.e.budget-cost)
            cost, conv = cost+dc, conv+dv
        progress = (env.t+h)/len(env.e.slots)
        # Soft pacing regularizer: plan ahead instead of exhausting the day early.
        score = env.utility(cost, conv)-.15*env.e.budget/env.e.cpa*(cost/env.e.budget-progress)**2
        if score > best:
            best, best_action = score, actions[0]
    return float(best_action)

def trajectory(e, label, penalty=1.):
    env = Replay(e, penalty)
    states, actions, rewards = [], [], []
    while env.t < len(e.slots):
        states.append(env.state())
        a = mpc(env) if label == 'mpc' else 1.
        actions.append([a])
        rewards.append(env.step(a))
    rtg = np.cumsum(rewards[::-1])[::-1].copy()
    return dict(states=np.array(states), actions=np.array(actions, dtype=np.float32),
                rtg=rtg.astype(np.float32)[:, None], rewards=np.array(rewards))

class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))

    def forward(self, x, causal, padding, inspect=False):
        z = self.ln1(x)
        z, weights = self.attn(z, z, z, attn_mask=causal, key_padding_mask=padding,
                               need_weights=inspect, average_attn_weights=False)
        x = x+z
        return x+self.ff(self.ln2(x)), weights

class DT(nn.Module):
    def __init__(self, width=64, layers=2, heads=4):
        super().__init__()
        self.config = dict(width=width, layers=layers, heads=heads)
        self.state = nn.Linear(len(FEATURES), width)
        self.action, self.rtg = nn.Linear(1, width), nn.Linear(1, width)
        self.time = nn.Embedding(512, width)
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.head = nn.Linear(width, 1)

    def forward(self, s, a, r, t, valid=None, inspect=False):
        b, k = s.shape[:2]
        tm = self.time(t)
        x = torch.stack((self.rtg(r)+tm, self.state(s)+tm, self.action(a)+tm), 2)
        x = self.norm(x.reshape(b, k*3, -1))
        causal = torch.ones(k*3, k*3, device=x.device, dtype=torch.bool).triu(1)
        padding = ~valid.repeat_interleave(3, 1) if valid is not None else None
        maps = []
        for block in self.blocks:
            x, weights = block(x, causal, padding, inspect)
            if inspect:
                maps.append(weights)
        # s_t precedes a_t; its output cannot read its own target action.
        out = 2*torch.sigmoid(self.head(x.reshape(b, k, 3, -1)[:, :, 1]))
        return (out, maps) if inspect else out

class Policy:
    def __init__(self, checkpoint, target=None):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['schema'] != SCHEMA or ck['features'] != FEATURES:
            raise ValueError('Feature schema mismatch')
        self.model = DT(**ck['config']).eval()
        self.model.load_state_dict(ck['model'])
        self.mean, self.std = np.array(ck['mean']), np.array(ck['std'])
        self.scale, self.context = ck['return_scale'], ck['context']
        self.penalty = ck['penalty']
        self.target = ck['target'] if target is None else target
        self.reset()

    def reset(self):
        self.states, self.actions, self.returns, self.times = [], [], [], []
        self.remaining = self.target

    def act(self, state, timestep, previous_reward=0.):
        self.remaining -= previous_reward
        self.states.append((state-self.mean)/self.std)
        self.actions.append([0.])
        self.returns.append([self.remaining/self.scale])
        self.times.append(timestep)
        def tensor(x, dtype=torch.float32):
            return torch.tensor(np.array(x[-self.context:]), dtype=dtype)[None]
        with torch.no_grad():
            a = self.model(tensor(self.states), tensor(self.actions), tensor(self.returns),
                           tensor(self.times, torch.long))[0, -1, 0].item()
        self.actions[-1] = [a]
        return a

def batch(trajectories, mean, std, scale, context, size, rng, device):
    s = np.zeros((size, context, len(FEATURES)), np.float32)
    a, r = np.zeros((size, context, 1), np.float32), np.zeros((size, context, 1), np.float32)
    t, valid = np.zeros((size, context), np.int64), np.zeros((size, context), bool)
    for i in range(size):
        tr = trajectories[rng.integers(len(trajectories))]
        start = rng.integers(len(tr['states']))
        n = min(context, len(tr['states'])-start)
        sl = slice(start, start+n)
        s[i, :n] = (tr['states'][sl]-mean)/std
        a[i, :n], r[i, :n] = tr['actions'][sl], tr['rtg'][sl]/scale
        t[i, :n], valid[i, :n] = np.arange(start, start+n), True
    return [torch.tensor(x, device=device) for x in (s, a, r, t, valid)]

def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    eps = read_csv(args.csv) if args.csv else synthetic(args.seed)
    periods = sorted({e.key.split(':')[0] for e in eps}, key=int)
    if len(periods) < 3:
        raise ValueError('At least 3 independent periods are required')
    cut1, cut2 = max(1, int(len(periods)*.6)), max(2, int(len(periods)*.8))
    groups = [set(periods[:cut1]), set(periods[cut1:cut2]), set(periods[cut2:])]
    train_eps, val_eps, test_eps = [[e for e in eps if e.key.split(':')[0] in g] for g in groups]
    fingerprint = hashlib.sha256()
    for e in train_eps+val_eps:
        fingerprint.update(f'{e.key}:{e.budget}:{e.cpa}'.encode())
        for pvalues, prices in e.slots:
            fingerprint.update(pvalues.tobytes())
            fingerprint.update(prices.tobytes())
    fingerprint.update(f'{SCHEMA}:{args.label}:{args.penalty}:mpc2-v1'.encode())
    data_hash = fingerprint.hexdigest()
    cache = out/'trajectories.pt'
    if cache.exists():
        stored = torch.load(cache, weights_only=False)
        if stored['hash'] != data_hash:
            raise ValueError('Output directory contains a different dataset/config. Use a new --output.')
        train, val = stored['train'], stored['val']
    else:
        print('Preparing hindsight teacher trajectories on CPU...', flush=True)
        train = [trajectory(e, args.label, args.penalty) for e in train_eps]
        val = [trajectory(e, args.label, args.penalty) for e in val_eps]
        torch.save(dict(hash=data_hash, train=train, val=val), cache)
    states = np.concatenate([tr['states'] for tr in train])
    mean, std = states.mean(0), np.maximum(states.std(0), .01)
    scale = max(float(np.std([tr['rtg'][0, 0] for tr in train])), 1.)
    target = float(np.quantile([tr['rtg'][0, 0] for tr in train], .8))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install a CUDA PyTorch wheel or use --device cpu')
    model = DT(width=args.width, layers=args.layers, heads=args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best = [], float('inf')
    val_rng = np.random.default_rng(1000+args.seed)
    vb = batch(val, mean, std, scale, args.context, 64, val_rng, device)
    initial, start = None, 0
    signature = dict(data_hash=data_hash, config=model.config, context=args.context,
                     batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    if args.resume:
        saved = torch.load(out/'last.pt', map_location=device, weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Resume configuration/dataset mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        history, best, initial = saved['history'], saved['best'], saved['initial']
        start = saved['step']+1
    (out/'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf8')
    train_loss = None
    for step in range(start, args.updates+1):
        if step > 0:
            model.train()
            s, a, r, t, mask = batch(train, mean, std, scale, args.context, args.batch_size, rng, device)
            pred = model(s, a, r, t, mask)
            loss = (pred-a).square()[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_loss = float(loss.detach())
        if step % args.log_every == 0 or step == args.updates:
            model.eval()
            with torch.no_grad():
                s, a, r, t, mask = vb
                val_loss = float((model(s, a, r, t, mask)-a).square()[mask].mean())
            initial = val_loss if initial is None else initial
            history.append(dict(step=step, train_action_mse=train_loss, val_action_mse=val_loss))
            print(json.dumps(history[-1]), flush=True)
            if val_loss < best:
                best = val_loss
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}, config=model.config,
                                schema=SCHEMA, features=FEATURES, mean=mean.tolist(), std=std.tolist(),
                                return_scale=scale, target=target, context=args.context,
                                penalty=args.penalty, label=args.label), out/'model.pt')
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), signature=signature,
                            step=step, history=history, best=best, initial=initial,
                            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None), out/'last.pt')
            (out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf8')
    # Export one concrete validation trajectory for reading state/action/reward/RTG.
    example = val[0]
    detail = pd.DataFrame(example['states'], columns=FEATURES)
    detail['target_bid_multiplier'] = example['actions'][:,0]
    detail['reward'] = example['rewards']
    detail['rtg'] = example['rtg'][:,0]
    detail.to_csv(out/'teacher_example.csv', index_label='timestep')
    s,a,r,t,mask = vb
    best_model = Policy(out/'model.pt').model.to(device)
    with torch.no_grad():
        predictions = best_model(s,a,r,t,mask)
    pd.DataFrame({'target': a[mask].cpu().numpy().ravel(),
                  'prediction': predictions[mask].cpu().numpy().ravel()}).to_csv(out/'pred_vs_target.csv', index=False)
    policy = Policy(out/'model.pt')
    rows, step_rows = [], []
    for e in test_eps:
        for name in ['fixed', 'mpc', 'dt_rtg_0.5', 'dt_rtg_1.0', 'dt_rtg_1.5']:
            env = Replay(e, args.penalty)
            policy.target = target*float(name.rsplit('_', 1)[1]) if name.startswith('dt') else target
            policy.reset()
            reward, actions = 0., []
            while env.t < len(e.slots):
                state = env.state()
                a = policy.act(env.state(), env.t, reward) if name.startswith('dt') else (mpc(env) if name == 'mpc' else 1.)
                actions.append(a)
                reward = env.step(a)
                if e == test_eps[0]:
                    step_rows.append(dict(episode=e.key, policy=name, timestep=env.t-1,
                                          action=a, reward=reward, cumulative_cost=env.cost,
                                          cumulative_conversions=env.conv,
                                          desired_rtg=policy.returns[-1][0]*scale if name.startswith('dt') else None,
                                          **dict(zip(FEATURES, state.tolist()))))
            rows.append(dict(episode=e.key, policy=name, conversions=env.conv, cost=env.cost,
                             cpa=env.cost/max(env.conv, 1e-9), cpa_violation=env.cost>e.cpa*env.conv+1e-8,
                             utility=env.utility(env.cost, env.conv), budget_usage=env.cost/e.budget,
                             action_mean=float(np.mean(actions))))
    pd.DataFrame(rows).to_csv(out/'cases.csv', index=False)
    pd.DataFrame(step_rows).to_csv(out/'rollout_example.csv', index=False)
    summary = pd.DataFrame(rows).groupby('policy').mean(numeric_only=True).reset_index().to_dict('records')
    report = dict(data='public_csv' if args.csv else 'synthetic', seed=args.seed,
                  split={n:[e.key for e in es] for n, es in zip(['train','validation','test'], [train_eps,val_eps,test_eps])},
                  label=args.label, penalty=args.penalty, initial_val_mse=initial, best_val_mse=best,
                  history=history, metrics=summary)
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    # Temporal attention is diagnostic, not feature-level importance or causality.
    policy = Policy(out/'model.pt')
    tr = val[0]
    n = min(args.context, len(tr['states']))
    with torch.no_grad():
        _, maps = policy.model(torch.tensor((tr['states'][:n]-mean)/std)[None],
                              torch.tensor(tr['actions'][:n])[None], torch.tensor(tr['rtg'][:n]/scale)[None],
                              torch.arange(n)[None], inspect=True)
    np.savez_compressed(out/'attention.npz', **{f'layer_{i}': x.numpy() for i,x in enumerate(maps)})
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', nargs='+')
    p.add_argument('--output', default='runs/demo')
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--context', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--label', choices=['mpc','fixed'], default='mpc')
    p.add_argument('--penalty', type=float, default=1.)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.updates < 0 or not 1 <= args.context <= 48 or args.penalty < 0:
        p.error('updates >= 0, context in [1,48], penalty >= 0 required')
    if min(args.batch_size,args.width,args.layers,args.heads,args.log_every) <= 0 or args.width % args.heads or args.lr <= 0:
        p.error('Positive batch/model/log/lr required, width must be divisible by heads')
    run(args)
