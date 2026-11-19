"""Synthetic interface/boundary tests; does NOT substitute for robosuite testing."""
from pathlib import Path
import tempfile
import numpy as np
import torch
import pytest

from worldmodel_stage import (OBS_DIM, ACTION_DIM, load_episode, split_episodes,
                              sample_batch, fit_stats, device_stats,
                              DefenseRSSM, openloop_report, world_positions)

def make_demo_files(directory, n=10, length=84, seed=1):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    for ep in range(n):
        states=np.zeros((length+1,60),dtype="float32")
        actions=rng.uniform(-.4,.4,(length,7)).astype("float32")
        for t in range(length+1):
            ee=np.array([.03*np.sin(.03*t),.02*np.cos(.03*t),.94],dtype="float32")
            states[t,14:17]=ee
            states[t,17:20]=[.02,0,0]
            states[t,0:7]=np.arange(7)*.01
            states[t,56:60]=[1,0,0,0]
            for j in range(5):
                initial_angle=2*np.pi*j/5
                pos=np.array([(.38-.00045*t)*np.cos(initial_angle)+.01*ep/10,
                              (.38-.00045*t)*np.sin(initial_angle)+
                               .012*np.sin(.07*t+.2*j+.02*ep), .822])
                states[t,20+6*j:23+6*j] = pos-ee
                states[t,23+6*j:26+6*j] = [1,0,0]
            states[t,50:55]=0
            states[t,55]=t/1400
        terminated = bool(ep%2==0)
        if terminated:
            states[-1,50]=2
            states[-1,23]=0
        disc=np.zeros(length,dtype="float32")
        if terminated: disc[-1]=-12
        cont=np.ones(length,dtype="float32")
        if terminated: cont[-1]=0
        done=np.zeros(length,dtype=bool);done[-1]=True
        terminated_vec=np.zeros(length,dtype=bool)
        truncated_vec=np.zeros(length,dtype=bool)
        terminated_vec[-1]=terminated
        truncated_vec[-1]=not terminated
        is_first=np.zeros(length,dtype=bool);is_first[0]=True
        is_last=np.zeros(length,dtype=bool);is_last[-1]=True
        np.savez_compressed(directory/f"episode_{ep:05d}.npz",
            obs=states[:-1],next_obs=states[1:],action=actions,
            reward=disc-.1,r_diff=-.1*np.ones(length,dtype="float32"),
            r_disc=disc,continuation=cont,done=done,
            terminated=terminated_vec,truncated=truncated_vec,
            is_first=is_first,is_last=is_last)


def test_shape_split_sampling(tmp_path):
    make_demo_files(tmp_path,n=9,length=44)
    paths=sorted(tmp_path.glob("*.npz"))
    a,b,c=split_episodes(paths)
    assert set(a).isdisjoint(b) and set(a).isdisjoint(c) and set(b).isdisjoint(c)
    assert len(a)+len(b)+len(c)==9
    ep=load_episode(paths[0])
    assert ep["states"].shape==(45,60)
    assert world_positions(ep["states"]).shape==(45,5,3)
    batch=sample_batch([ep],3,32,np.random.default_rng(99))
    assert batch["states"].shape==(3,33,60)
    assert batch["actions"].shape==(3,32,7)
    assert torch.allclose(batch["states"][:,1:,:],
        torch.stack([batch["states"][i,1:,:] for i in range(3)]))
    with pytest.raises(ValueError):
        sample_batch([ep],2,200,np.random.default_rng(1))


def test_model_train_and_forecast_no_future_obs(tmp_path):
    torch.manual_seed(1)
    torch.set_num_threads(1)
    make_demo_files(tmp_path,n=6,length=64)
    episodes=[load_episode(p) for p in sorted(tmp_path.glob("*.npz"))]
    stats=device_stats(fit_stats(episodes[:4]),"cpu")
    model=DefenseRSSM(deter=24,latent=8,embed=32)
    batch=sample_batch(episodes[:4],2,10,np.random.default_rng(1))
    model.train()
    loss,parts=model.batch_loss(batch,stats)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.rnn.weight_hh.grad is not None
    assert torch.isfinite(model.rnn.weight_hh.grad).all()
    model.eval()
    c=torch.from_numpy(episodes[4]["states"][:21])[None]
    aa=torch.from_numpy(episodes[4]["action"][:20])[None]
    af=torch.from_numpy(episodes[4]["action"][20:30])[None]
    pred, sp, rw, cp = model.forecast(c,aa,af,stats)
    assert pred.shape==(1,10,5,3) and torch.isfinite(pred).all()
    assert sp.shape==(1,10,5,3) and torch.allclose(sp.sum(-1),torch.ones(1,10,5))
    assert rw.shape==(1,10) and cp.shape==(1,10)
    # Future states are never passed to forecast, so cannot leak.
    modified=episodes[4]["states"].copy()
    modified[21:] = -999
    assert torch.equal(pred,model.forecast(c,aa,af,stats)[0])


def test_openloop_lengths_and_invalid_fields(tmp_path):
    make_demo_files(tmp_path,n=4,length=65)
    paths=sorted(tmp_path.glob("*.npz"))
    ep=load_episode(paths[0])
    model=DefenseRSSM(deter=16,latent=5,embed=24)
    stats=device_stats(fit_stats([ep]),"cpu")
    results=openloop_report(model,[ep],stats,burn=15,
                            horizons=(10,20,40),windows_per_ep=1)
    assert all(results[str(h)]["windows"]==1 for h in (10,20,40))
    assert all(results[str(h)]["wm_m"] >= 0 for h in (10,20,40))
    # Boundary: forecast horizon longer than episode => nothing evaluated.
    short=openloop_report(model,[ep],stats,burn=60,horizons=(40,),windows_per_ep=1)
    assert short["40"]["windows"]==0 and short["40"]["wm_m"] is None
    with np.load(paths[1]) as f:
        d={k:f[k].copy() for k in f.files}
    d["continuation"][-1]=0.  # episode is time-limit truncated
    bad=tmp_path/"bad.npz"
    np.savez_compressed(bad,**d)
    with pytest.raises(ValueError, match="continuation"):
        load_episode(bad)
    d["continuation"][-1]=1.
    d["obs"][0,0]=np.nan
    np.savez_compressed(bad,**d)
    with pytest.raises(ValueError, match="nonfinite"):
        load_episode(bad)


def test_three_episode_split():
    v=list(range(3))
    train,val,test=split_episodes(v,9)
    assert sorted(train+val+test)==v
    with pytest.raises(ValueError):
        split_episodes([1,2])


def test_posterior_filter_and_bad_shapes(tmp_path):
    from integration_reference import LatentFilter
    make_demo_files(tmp_path,n=3,length=40)
    episodes=[load_episode(p) for p in sorted(tmp_path.glob("*.npz"))]
    model=DefenseRSSM(deter=20,latent=7,embed=24)
    filt=LatentFilter(model,fit_stats(episodes),"cpu")
    with pytest.raises(RuntimeError):
        filt.update(np.zeros(7),np.zeros(60))
    z0=filt.reset(episodes[0]["obs"][0])
    assert z0.shape==(1,27) and not z0.requires_grad
    z1=filt.update(episodes[0]["action"][0],episodes[0]["next_obs"][0])
    assert z1.shape==(1,27) and torch.isfinite(z1).all()
    with pytest.raises(ValueError,match="60"):
        filt.reset(np.zeros(61))
    with pytest.raises(ValueError,match="7"):
        filt.update(np.zeros(5),episodes[0]["next_obs"][1])
