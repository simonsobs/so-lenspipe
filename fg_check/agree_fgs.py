import argparse
from solenspipe import foregrounds as fg
import numpy as np

# Draw the pa5a foreground alms for cmb_set 0, sim 3, check they are identical to
# the reference draw from the main pipeline (--ref), and save them (--out).
parser = argparse.ArgumentParser(description='check foreground realizations agree')
parser.add_argument("--out", type=str, default='my_fg_pa5a_set0_sim3.npy',
                    help='where to save the foreground alms')
parser.add_argument("--ref", type=str,
                    default='/scratch/iabril/20261013/fg_check/fg_pa5a_set0_sim3.npy',
                    help='reference .npy to compare against (main pipeline draw)')
args = parser.parse_args()

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

x = fg_handler.get_fg_alms(fgcov, 'pa5a', 0, 3)

if args.ref is not None:
    ref = np.load(args.ref)
    print(f'identical to {args.ref}:', np.array_equal(x, ref))

np.save(args.out, x)
print(f'saved {args.out}')