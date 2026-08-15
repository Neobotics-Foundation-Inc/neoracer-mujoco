# Vendored fixture: Ultimate Wall Follower

`control.py`, `wall_follower.py`, and `config/wall_follower.yaml` in this
directory are byte-for-byte copies from:

- Repo: https://github.com/Neobotics-Foundation-Inc/neoracer-labs
- Path: `ultimate-wall-follower/`
- Commit: `07b3f1c3994835b47957879e6abe704e334664ad`

Used by `validation/test_wall_follower_example.py` to prove (issue #28) that
this canonical racecar_core controller runs against `MujocoRacecar` with no
changes to its logic. Do not edit these three files -- if the upstream
example changes, re-vendor them from neoracer-labs instead of patching here.
