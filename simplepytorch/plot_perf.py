#!/usr/bin/env -S ipython --no-banner -i --
"""
Quick performance plots used as a script to visualize results.

$ ipython -im simplepytorch.perf_plot -- -h

I've aliased this in my bash shell to

alias sdp='ipython -i -m simplepytorch.perf_plot -- '
"""
import argparse as ap
import datetime as dt
import sys
import re
import glob
import os
from os.path import join, exists, basename, dirname
#  from os import exists
import pandas as pd
import seaborn as sns
from textwrap import dedent
from matplotlib import pyplot as plt
#  import mpld3
#  import mpld3.plugins


def get_run_ids(ns):
    dirs = glob.glob(join(ns.data_dir, '*'))
    for dir in dirs:
        run_id = basename(dir)
        if not re.search(ns.runid_regex, basename(dir)):
            continue
        yield run_id


def load_df_from_fp(fp, ns):
    print('found data', fp)
    try:
        if fp.endswith('.csv'):
            df = pd.read_csv(fp).set_index(ns.index_col)
        elif fp.endswith('.h5'):
            df = pd.read_hdf(fp, ns.hdf_table)
    except:
        print(f"   WARNING: failed to load {fp}")
        df = pd.DataFrame()
    return df


def _mode1_get_frames(ns):
    for run_id in get_run_ids(ns):
        #  if not exists(join(dir, 'lock.finished')):
            #  continue
        dirp = f'{ns.data_dir}/{run_id}'
        gen = (
            ((run_id, fname), load_df_from_fp(join(dirp, fname), ns))
            for fname in os.listdir(dirp)
            if re.search(ns.data_fp_regex, fname))
        yield from (x for x in gen if not x[-1].empty)


def make_plots(ns, cdfs):
    plot_cols = [col for col in cdfs.columns if re.search(ns.col_regex, col)]
    for col in plot_cols:
        fig, ax = plt.subplots(1,1)
        df = cdfs[col].unstack(level=0)
        if ns.rolling_mean:
            df = df.rolling(ns.rolling_mean).mean()
        _plot_lines = df.plot(ax=ax, title=col)
        #  _legend = mpld3.plugins.InteractiveLegendPlugin(
            #  *_plot_lines.get_legend_handles_labels())
        #  mpld3.plugins.connect(fig, _legend)

        yield (fig, ax, col)


def savefig_with_symlink(fig, fp, symlink_fp):
    os.makedirs(dirname(fp), exist_ok=True)
    fig.savefig(fp, bbox_inches='tight')
    if os.path.islink(symlink_fp):
        os.remove(symlink_fp)
    prefix = os.path.dirname(os.path.commonprefix([fp, symlink_fp]))
    os.symlink(fp[len(prefix)+1:], symlink_fp)
    print('save fig', symlink_fp)


def main(ns):
    print(ns)
    # mode 1: compare each column across all files
    if ns.mode == 1:
        mode_1_plots(ns)
    elif ns.mode == 2:
        mode_2_plots(ns)
    elif ns.mode == 3:
        mode_3_plots(ns)
    else:
        raise Exception(f'not implemented mode: {ns.mode}')
    if ns.no_plot:
        print("type 'plt.show()' in terminal to see result")
    else:
        plt.show(block=False)


def _mode_1_get_perf_data_as_df(ns):
    dfs = dict(_mode1_get_frames(ns))
    cdfs = pd.concat(dfs, sort=False, names=['run_id', 'filename'])
    return cdfs


def mode_1_plots(ns):
    """Compare experiments.  
    One plot for each metric, comparing the most recent result of each experiment
    """
    cdfs = _mode_1_get_perf_data_as_df(ns).reset_index('filename', drop=True)
    globals().update({'cdfs_mode1': cdfs})

    timestamp = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')  # date plot was created.  nothing to do with timestamp column.
    os.makedirs(join(ns.mode1_savefig_dir, 'archive'), exist_ok=True)
    for fig, ax, col in make_plots(ns, cdfs):
        if not ns.savefig: continue
        savefig_with_symlink(
            fig,
            f'{ns.mode1_savefig_dir}/archive/{col}_{timestamp}.png',
            f'{ns.mode1_savefig_dir}/{col}_latest.png')


def mode_2_plots(ns):
    """Analyze an experiment's historical performance across many runs.
    One plot for each experiment and metric, to analyze history of runs for
    that experiment"""
    # mode 2: compare train to val performance
    cdfs_mode2 = {}
    timestamp = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')  # date plot was created.  nothing to do with timestamp column.
    for run_id in get_run_ids(ns):
        dirp = f'{ns.data_dir}/{run_id}/log'
        if not os.path.exists(dirp):
            print('skip', run_id, 'contains no log data')
            continue

        cdfs = pd.concat({
            (run_id, fname): load_df_from_fp(join(dirp, fname), ns)
            for fname in os.listdir(dirp)
            if re.search(f'{ns.data_fp_regex}', fname)},
            sort=False, names=['run_id', 'filename']
        )
        cdfs_mode2[run_id] = cdfs

        os.makedirs(f'{ns.mode2_savefig_dir}/archive'.format(run_id=run_id), exist_ok=True)
        for fig, ax, col in make_plots(ns, cdfs.reset_index('run_id', drop=True)):
            fig.suptitle(run_id)
            if not ns.savefig: continue
            # save to file
            savefig_with_symlink(
                fig,
                f'{ns.mode2_savefig_dir}/archive/{col}_{timestamp}.png'.format(run_id=run_id),
                f'{ns.mode2_savefig_dir}/{col}_latest.png'.format(run_id=run_id))
    globals().update({'cdfs_mode2': cdfs_mode2})


def mode_3_plots(ns):
    """Compare across experiments, considering their history of runs.
    Basically combines mode 1 and mode 2. """
    cdfs_mode3 = {}
    timestamp = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')  # date plot was created.  nothing to do with timestamp column.
    for run_id in get_run_ids(ns):
        dirp = f'{ns.data_dir}/{run_id}/log'
        if not os.path.exists(dirp):
            print('skip', run_id, 'contains no log data')
            continue

        cdfs = pd.concat({
            (run_id, fname): load_df_from_fp(join(dirp, fname), ns)
            for fname in os.listdir(dirp)
            if re.search(f'{ns.data_fp_regex}', fname)},
            sort=False, names=['run_id', 'filename']
        )
        cdfs_mode3[run_id] = cdfs
    cdfs = pd.concat(cdfs_mode3.values())

    #  cdfs.groupby(['run_id', 'epoch']).agg(ns.mode3_agg_method)
    cdfs_unmodified = cdfs.copy()
    if ns.rolling_mean:
        cdfs = cdfs.rolling(ns.rolling_mean).mean()

    plot_cols = [col for col in cdfs.columns if re.search(ns.col_regex, col)]
    for col in plot_cols:
        fig, ax = plt.subplots(1,1, figsize=(12,10))
        sns.lineplot(x='epoch', y=col, hue='run_id', ax=ax, data=cdfs)
        ax.set_title(col)
        if not ns.savefig: continue
        savefig_with_symlink(
            fig,
            f'{ns.mode3_savefig_dir}/archive/{col}_mode3_{timestamp}.png',
            f'{ns.mode3_savefig_dir}/{col}_mode3_latest.png')
    globals().update({'cdfs_mode3': cdfs_unmodified})


def bap():
    class F(ap.ArgumentDefaultsHelpFormatter, ap.RawTextHelpFormatter): pass
    par = ap.ArgumentParser(formatter_class=F)
    A = par.add_argument
    A('runid_regex', help='find the run_ids that match the regular expression.')
    A('--data-dir', default='./results', help=' ')
    A('--data-fp-regex', default='perf.*\.csv', help=' ')
    #  A('--', nargs='?', default='one', const='two')
    A('--rolling-mean', '--rm', type=int, help='smoothing')
    A('--hdf-table-name', help='required if searching .h5 files')
    A('-c', '--col-regex', default='^(?!epoch|batch_idx|timestamp)', help='plot only columns matching regex.  By default, plot all except epoch and batch_idx.')
    A('--index-col', default='epoch', help=' ')
    A('--mode', default=1, type=int, choices=[1,2,3], help=dedent('''\
        `--mode 1` compare across experiments, with one plot per column (i.e. performance metric)
        `--mode 2` Within one experiment and column, visualize the history of all runs.
        `--mode 3` combine 1 and 2.  compare across run_ids, with
                   confidence interval to consider history of runs for each run_id.'''))
    A('--mode1-savefig-dir', default='./results/plots/mode1', help=' ')
    A('--mode2-savefig-dir', default='./results/plots/mode2', help=' ')
    A('--mode3-savefig-dir', default='./results/plots/mode3', help=' ')
    A('--best-effort', action='store_true', help=" Try to load a csv file, but don't raise error if cannot read a file.")
    A('--savefig', action='store_true', help="save plots to file")
    A('--no-plot', action='store_true', help="if supplied, don't show the plots")
    return par


if __name__ == "__main__":
    main(bap().parse_args())
