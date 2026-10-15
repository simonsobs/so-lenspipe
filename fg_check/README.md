# Adding the ACT+Planck extragalactic foregrounds to your sims

These steps give you the same foreground realization as the main DR6+ lensing sims for every `(cmb_set, sim_id)`.

## 1. Code version

You don't need to switch to the `brexit` branch. Copy just two things from it: the module `solenspipe/foregrounds.py` and this `fg_check/` folder. From your so-lenspipe checkout on `preproc_test`:

```bash
git fetch origin brexit
git checkout origin/brexit -- solenspipe/foregrounds.py fg_check
```

This adds `solenspipe/foregrounds.py` and `fg_check/`. `preproc_test` has neither, so nothing on your branch is overwritten. Everything it imports is already on `preproc_test`. If your install isn't editable (`pip install -e .`), reinstall after copying the file so Python picks it up.

**Switch your import.** On `preproc_test`, `ForegroundHandler` lives in `solenspipe/preprocessing.py`. That copy is the old version: it has no day-time copies and no seed-keyed redraw, and it gives different foregrounds. Replace

```python
from solenspipe import preprocessing as prep
foreground_handler = prep.ForegroundHandler(datamodel, args)
```

with

```python
from solenspipe import foregrounds as fg
foreground_handler = fg.ForegroundHandler(datamodel, args)
```

and keep using `prep` for everything else.

If you can, use the same pixell version as the main runs too, since the random draw comes from pixell's `rand_alm`.

## 2. What must match exactly

| | Value | Why |
|---|---|---|
| Foreground model | `/project/rrg-rbond-ac/msyriac/fgcovcubes/20261008/fg_fsky_70.npz` | Mat's ACT+Planck fit (fsky 70%) |
| fg qids | `pa5a pa5b pa6a pa6b p04 p05 p06 p07 p08`, same order | The realization is drawn jointly for all of these, so a different list or order changes every array's foregrounds |
| Seed | `(cmb_set, sim_id)` | Used as `seed=(0, cmb_set, sim_id)`. Both numbers matter: the same `sim_id` in a different `cmb_set` is a different sky |
| `lmax_signal` | 5100 (same as the main runs) | Sets the size of the draw |

**About `sim_id`:** use the plain sim index, the one in the saved file names. Don't use the index you get from `convert_seeds` for the CMB alms. For `cmb_set=0` the two are the same. For sets 1–3 the converted one is offset by 500 or 1000, so passing it gives a different foreground sky.

## 3. Code

```python
import argparse
from solenspipe import foregrounds as fg

FG_PATH  = '/project/rrg-rbond-ac/msyriac/fgcovcubes/20261008/fg_fsky_70.npz'
FG_QIDS  = 'pa5a pa5b pa6a pa6b p04 p05 p06 p07 p08'   # do not change or reorder
LMAX_SIG = 5100                                         # must match the main runs

fg_args = argparse.Namespace(
    fg_type='sims_actplanck',
    fgs_path=FG_PATH,
    fg_qids=FG_QIDS,
    qids=FG_QIDS,          # not used for the draw when fg_qids is set
    lmax_signal=LMAX_SIG,
)
fg_handler = fg.ForegroundHandler(datamodel=None, args=fg_args)
fgcov = fg_handler.fgcov_func()      # load once, outside the sim loop

for sim_id in my_sim_ids:            # plain sim index (as in the file names), not the converted CMB index
    cmb_alms = ...                   # your (T, E, B) signal alms, complex128, lmax = LMAX_SIG
    for qid in my_qids:              # e.g. 'pa5a', 'pa6b_dw', 'p05'
        fg_alms = fg_handler.get_fg_alms(fgcov, qid, cmb_set, sim_id)
        signal_alms = cmb_alms.copy()
        signal_alms[0] += fg_alms    # temperature only
        # ...then beam-convolve and process signal_alms as you do for the CMB
```

If you use `DR6plus_lensing/scripts/kspace_process_data_sims.py` directly, don't write any of this. Pass these flags instead:

```
--is-sim --fg-type sims_actplanck \
--fgs-path /project/rrg-rbond-ac/msyriac/fgcovcubes/20261008/fg_fsky_70.npz \
--fg-qids "pa5a pa5b pa6a pa6b p04 p05 p06 p07 p08" \
--lmax-signal 5100 --cmb-set <set>
```

## 4. How it behaves

- **One draw per sim:** the joint realization is drawn once for each `(cmb_set, sim_id)` and reused for every qid in that sim. Just pass the seed on every call; you don't need `rerun`.
- **Seed is required:** leaving out `cmb_set` or `sim_id` raises a `ValueError`. There are no unseeded draws.
- **Day-time arrays:** a qid ending in `_dw` or `_dd` gets an exact copy of its night-time array's foregrounds (`pa5a_dw` gets `pa5a`'s), meaning the same foreground sky.
- **Supported qids:** each qid you process (after stripping `_dw`/`_dd`) must be in the fg qids list, or you get a `ValueError`. `pa4b` and `p03` are in the model but not in the list. Adding them would change everyone's realization, so check with the main pipeline before doing that.
- **Where to add them:** the foregrounds go into the T alms before beam convolution, together with the CMB signal.

## 5. Check that it matches

Run `fg_check/agree_fgs.py` from the root of your so-lenspipe checkout. It draws the `pa5a` foregrounds for `cmb_set=0`, `sim_id=3` with the settings above, and compares them with the main pipeline's draw. That reference is `/scratch/iabril/20261013/fg_check/fg_pa5a_set0_sim3.npy` (208 MB, too big for git), which members of the `def-rbond` group can read:

```bash
python fg_check/agree_fgs.py
```

It should print `identical to /scratch/iabril/20261013/fg_check/fg_pa5a_set0_sim3.npy: True`. It also saves your draw to `my_fg_pa5a_set0_sim3.npy` in the current directory. Use `--out` to save it somewhere else, and `--ref` to compare against a different file. If it prints `False`, check the four items in section 2 and that `solenspipe.foregrounds` is the copied file (`python -c "import solenspipe.foregrounds as f; print(f.__file__)"`).

This tests the copied `foregrounds.py` and its settings. It doesn't test your own script, so also check that your script calls `get_fg_alms` with `cmb_set` and the plain `sim_id`.
