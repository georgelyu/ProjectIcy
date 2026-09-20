"""Collect local SNIBB flat/curved grid studies without running a solver."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for case in ['flat', 'curved']:
        data = [json.loads((args.root/f'snibb_{case}_{n}'/'result.json').read_text())
                for n in [32, 64, 128]]
        errors = np.array([item['relative_velocity_l2'] for item in data])
        results[case] = {'runs': data, 'error_reduction': (errors[:-1]/errors[1:]).tolist(),
                        'observed_orders': np.log2(errors[:-1]/errors[1:]).tolist()}
        axes[0].loglog([32, 64, 128], errors, 'o-', label=case)
        with np.load(args.root/f'snibb_{case}_128'/'velocity.npz') as fields:
            error = np.linalg.norm(fields['velocity']-fields['exact'], axis=-1)/data[-1]['wall_speed_lattice']
            ax = axes[1 if case == 'flat' else 2]
            artist = ax.imshow(np.ma.array(error, mask=~fields['mask']).T, origin='lower', cmap='magma')
            ax.set_title(case+' velocity error / wall speed')
            fig.colorbar(artist, ax=ax, shrink=.8)
    axes[0].set(xlabel='Lattice width N', ylabel='Relative velocity L2 error',
                title='Local SNIBB Couette accuracy', xticks=[32, 64, 128], xticklabels=['32', '64', '128'])
    axes[0].minorticks_off()
    axes[0].legend()
    axes[0].grid(alpha=.2)
    fig.savefig(args.root/'snibb_accuracy.png', dpi=180)
    plt.close(fig)
    results['scope'] = ('Uniform-viscosity local pressure/MRT/SNIBB formulas; analytic signed distance. '
                        'Not full moving/melting cut-volume coupling or the material SDF proxy.')
    (args.root/'snibb_accuracy_report.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
