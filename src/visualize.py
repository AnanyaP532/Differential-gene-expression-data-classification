#plots and summary images from metrics produced by train_models.py
#python src/visualize.py --metrics results/metrics.json --outdir results/figures

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(metrics_path, outdir):
    with open(metrics_path) as f:
        metrics = json.load(f)
    df = pd.DataFrame(metrics).T
    df[['accuracy','precision','recall','f1']].plot.bar(rot=45)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'model_metrics_bar.png')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f'Wrote {outpath}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True, help='Metrics JSON file')
    parser.add_argument('--outdir', required=True, help='Output directory for figures')
    args = parser.parse_args()
    plot_metrics(args.metrics, args.outdir)
