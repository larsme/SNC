import os
import sys
import argparse
import matplotlib.pyplot as plt

from src.KittiDepthTrainer import *
from src.visualize_results import *

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', default=None, dest='mode',
                        help='"train", "predictions", "count_parameters", "results", "weights"')
    parser.add_argument('-params_dir', action='store', default=None, dest='params_dir',
                        help='Params file in workspace directory')
    parser.add_argument('-epoch', action='store', dest='epoch', default=-1, type=int, nargs='?',
                        help='Epoch to load checkpoint of')
    parser.add_argument('-sets', action='store', dest='sets', default=None, type=str, nargs='?',
                        help='Which sets to evaluate on "val", "selval" or "test"')
    parser.add_argument('-evaluate_all_epochs', default=False, type=bool)
    parser.add_argument('-save_chkpt_each', default=1, type=int)
    parser.add_argument('-runs', default=1, type=int)
    parser.add_argument('-run', default=-1, type=int)
    args = parser.parse_args()

    if args.sets is None:
        args.sets = ['train', 'val']

    if args.runs < 0:
        args.runs = 1
    if args.run < 0:
        args.run = 0

    if args.mode == 'results':
        visualize_results(**vars(args))
    else:
        max_run = args.run + args.runs - 1
        for run in range(args.run, max_run + 1):
            args.run = run

            trainer = KittiDepthTrainer(**vars(args))

            if args.mode == 'train':
                trainer.train(evaluate_all_epochs=args.evaluate_all_epochs)
            elif args.mode == 'predictions':
                trainer.display_predictions()
            elif args.mode == 'count_parameters':
                trainer.count_parameters()
            elif args.mode == 'weights':
                trainer.visualize_weights()
                if run == max_run:
                    plt.show()