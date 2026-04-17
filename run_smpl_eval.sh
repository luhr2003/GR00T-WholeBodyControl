#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv_isaac/bin/activate

rm -rf /tmp/sonic_smpl_eval
mkdir -p /tmp/sonic_smpl_eval

python gear_sonic/eval_agent_trl.py \
    +checkpoint=sonic_release/last.pt \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=True \
    ++num_envs=8 \
    +manager_env/terminations=tracking/eval \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
    ++manager_env.commands.motion.encoder_sample_probs.g1=0 \
    ++manager_env.commands.motion.encoder_sample_probs.teleop=0 \
    ++manager_env.commands.motion.encoder_sample_probs.smpl=1 \
    ++manager_env.config.render_results=True \
    ++manager_env.config.save_rendering_dir=/tmp/sonic_smpl_eval \
    ~manager_env/recorders=empty \
    +manager_env/recorders=render
