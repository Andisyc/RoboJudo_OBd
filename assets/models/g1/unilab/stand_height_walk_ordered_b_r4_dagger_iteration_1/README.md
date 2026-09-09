# UniLab stand-height/walk DAgger student

- Source repository: `/Users/sss9999/locomotion/unilab/unilab_stand`
- Source branch/commit: `codex/stand-height-walk-async` / `5fff4b33`
- Source checkpoint: `stand_height_walk_ordered_b_r4_dagger_iteration_1.pt`
- Source SHA-256: `4d9eed26b39a875a2475289f84a40949a95c3a7d5a618b7110f90df051fae6fe`
- Exported policy SHA-256: `c73c0bc17e20cc7644c54092ab131702e2ff05bbc3dc818f4715b5b9e940ab83`

The exported TorchScript accepts a `(batch, 99)` float observation and returns
a `(batch, 29)` residual joint action. It embeds UniLab's playback-time hard
routing contract: active planar/yaw commands select expert 0; inactive commands
select expert 1. Re-export with `scripts/export_unilab_distill.py` after replacing
the source checkpoint.
