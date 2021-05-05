import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import os

def visualize_results(params_dir=None, run=0, runs=1, sets=['val'], epoch=None, **args):
    for set in sets:
        metrics_file_pattern = '{}/*/metrics_{}.csv'.format(params_dir, set)
        num_exps = len(glob.glob(metrics_file_pattern))
        if num_exps == 0:
            metrics_file_pattern = '{}/metrics_{}.csv'.format(params_dir, set)
            num_exps = len(glob.glob(metrics_file_pattern))

        fig = plt.figure()
        plt.suptitle('{} on {} set'.format(params_dir.replace('workspace/',''), set))

        unified_df = pd.DataFrame()

        maxs = []
        mins = []
        delta_start = []
        subplots = []
        for metrics_file in glob.glob(metrics_file_pattern):
            exp = metrics_file.split('metrics')[0].replace('\\','/').split('/')[-2]
            metrics_df = pd.DataFrame.sort_index(pd.read_csv(metrics_file).set_index(['Epoch', 'Run']), level=['Epoch', 'Run'])
            means = metrics_df.mean(axis=0, level='Epoch')
            stds = metrics_df.std(axis=0, level='Epoch')

            idx = 0
            for metric in metrics_df.columns:
                n_subplots = len(metrics_df.columns) - 3
                rows = (n_subplots + 1) // 2
                if metric not in ['loss', 'Parameters', 'BatchDuration']:
                    if len(subplots) <= idx:
                        subplots.append(plt.subplot(rows, 2, (idx % rows) * 2 + (2 if idx >= rows else +1)))
                    else:
                        plt.subplot(subplots[idx])
                    plt.ylabel(metric)
                    plt.plot(means.index, means[metric], label = '{} ({} P, {:0.03} ms)'.format(exp, means['Parameters'][0], metrics_df['BatchDuration'].min() * 1000), lw=1)
                    plt.fill_between(means.index, means[metric] - stds[metric], means[metric] + stds[metric], color=plt.gca().lines[-1].get_color(), alpha=0.8 / num_exps)
                    idx+=1
                    if len(maxs) < idx:
                        maxs.append([])
                        mins.append([])
                        delta_start.append(0)
                    maxs[idx - 1].extend((means[metric] + stds[metric])[np.isfinite(stds[metric])].tolist())
                    mins[idx - 1].extend((means[metric] - stds[metric])[np.isfinite(stds[metric])].tolist())
                    maxs[idx - 1].extend((means[metric])[np.isnan(stds[metric])].tolist())
                    mins[idx - 1].extend((means[metric])[np.isnan(stds[metric])].tolist())
                    delta_start[idx - 1]+= list(means[metric])[-1] - list(means[metric])[0]

            metrics = {}
            for metric in metrics_df.columns:
                if metric in ['MAE', 'RMSE', 'wMAE', 'wRMSE']:
                    metrics[metric] = ['{:0.0f} mm ± {:0.0f}'.format(means.loc[ep][metric] * 1000, stds.loc[ep][metric] * 1000) for ep in means.index]
                elif metric == 'Parameters':
                    metrics[metric] = [str(int(means.loc[ep][metric])) for ep in means.index]
                elif 'Duration' in metric:
                    metrics[metric] = ['{:0.0f} ms ± {:0.0f}'.format(means.loc[ep][metric] * 1000, stds.loc[ep][metric] * 1000) for ep in means.index]
                elif 'Delta' in metric:
                    metrics[metric] = ['{:0.2%} ± {:0.2f}'.format(means.loc[ep][metric], stds.loc[ep][metric] * 100) for ep in means.index]
                else:
                    metrics[metric] = ['{:0.4f} ± {:0.4f}'.format(means.loc[ep][metric], stds.loc[ep][metric]) for ep in means.index]
            metrics['Exp'] = [exp for ep in means.index]
            metrics['Epoch'] = [ep for ep in means.index]

            unified_df = pd.concat((unified_df, pd.DataFrame(metrics, columns=list(metrics)).set_index(['Epoch', 'Exp'])))

        unified_df = pd.DataFrame.sort_index(unified_df, level=['Epoch', 'Exp'])
        if epoch is None:
            unified_df.to_csv('{}/unified_metrics.csv'.format(params_dir))
        else:
            try:
                unified_df = unified_df.loc[epoch]
                unified_df.to_csv('{}/unified_metrics_ep{:04.0f}.csv'.format(params_dir, epoch))
            except:
                print('epoch not in metrics of {} set:'.format(set))
        print(unified_df.to_markdown())
        if len(unified_df) > 0:
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')

            for idx in range(n_subplots):
                if len(subplots) <= idx:
                    ax = plt.subplot(rows, 2, (idx % rows) * 2 + (2 if idx >= rows else +1))
                else:
                    ax = plt.subplot(subplots[idx])
                if (idx < n_subplots - 1 and idx != rows - 1):
                    ax.set_xticklabels([])

                ax.yaxis.set_minor_locator(AutoMinorLocator())
                plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.3)
                plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.1)

                if delta_start[idx] > 0:
                    q1 = np.nanquantile(mins[idx], 0.1)
                    q2 = np.max(maxs[idx])
                    plt.ylim([q1 + 0.2 * (q1 - q2),  q2 + 0.1 * (q2 - q1)])
                else:
                    q1 = np.min(mins[idx])
                    q2 = np.nanquantile(maxs[idx], 0.9)
                    plt.ylim([q1 + 0.1 * (q1 - q2), q2 + 0.2 * (q2 - q1)])

                if epoch >= 0:
                    plt.xlim([0, epoch])

            subplots[n_subplots//2-1].legend(bbox_to_anchor=(-0.1,-0.2), loc='upper left', ncol=5)
            plt.savefig('{}/metrics_trajectories.png'.format(params_dir))
            plt.show()
        else:
            plt.close(fig)