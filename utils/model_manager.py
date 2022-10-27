"""
Utility functions to manipulate models in Weights and Baises

"""
import argparse

import wandb


def download_model_checkpoints(path):
    run = wandb.init()
    model_artifact = run.use_artifact(path, type='model')
    model_dir = model_artifact.download()
    print(f"model has been downloaded to {model_dir}")


def get_args():
    parser = argparse.ArgumentParser(description='Download or convert a model to script format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the location the model is stored in, in weights and biases")
    parser.add_argument('--script', '-s', action='store_true',
                        help='convert the model to a torchscript format')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.script:
        raise NotImplementedError("sorry that feature has not yet been implemented")

    download_model_checkpoints(path=args.model)
