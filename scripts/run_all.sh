#!/usr/bin/env bash
# Reproduce every number and figure in the report.
# Wall-clock on 32 CPU cores + one 24 GB GPU: ~1.2 h per PPO run, ~2.5 h per DQN run
# (the runs below are started in parallel; lower --num-envs if you have fewer cores).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STEPS=${STEPS:-5000000}

# 1) main agents: 3 seeds each
for s in 1 2 3; do
  python -m mining_rl.ppo --seed $s --total-steps $STEPS > logs/ppo_s$s.log 2>&1 &
done
for s in 1 2 3; do
  python -m mining_rl.dqn --seed $s --total-steps $STEPS > logs/dqn_s$s.log 2>&1 &
done
wait

# 2) ablations: reward shaping and observation design (2 seeds each)
for s in 1 2; do
  python -m mining_rl.ppo --seed $s --total-steps $STEPS --no-damage-penalty \
    --run-name ppo_nodmg_s$s > logs/ppo_nodmg_s$s.log 2>&1 &
  python -m mining_rl.ppo --seed $s --total-steps $STEPS --frame-stack 1 --no-inventory \
    --run-name ppo_pixels_s$s > logs/ppo_pixels_s$s.log 2>&1 &
done
wait
for s in 1 2; do
  python -m mining_rl.ppo --seed $s --total-steps $STEPS --frame-stack 1 \
    --run-name ppo_fs1_s$s > logs/ppo_fs1_s$s.log 2>&1 &
  python -m mining_rl.ppo --seed $s --total-steps $STEPS --no-inventory \
    --run-name ppo_noinv_s$s > logs/ppo_noinv_s$s.log 2>&1 &
done
wait

# 3) evaluation on held-out worlds (100 episodes per agent)
python -m mining_rl.evaluate --agent random --episodes 100
for ck in checkpoints/ppo_s?.pt checkpoints/dqn_s?.pt checkpoints/ppo_nodmg_s?.pt checkpoints/ppo_pixels_s?.pt \
          checkpoints/ppo_fs1_s?.pt checkpoints/ppo_noinv_s?.pt; do
  python -m mining_rl.evaluate --checkpoint "$ck" --episodes 100
done

# 4) figures + a GIF of the trained agent
python -m mining_rl.plot
python -m mining_rl.record --checkpoint checkpoints/ppo_s3.pt --out results/media/ppo_agent.gif
