"""Provide an image of handwritten text and get back out a string!"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import gradio as gr
from PIL import ImageStat
from PIL.Image import Image
from PIL import Image as PILIMAGE
import requests

from predict import LandcoverSegmentationClassifier
import utils.general as utils

STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "model-ckpt-epoch-90.pt:v0"
MODEL_FILE = "/UNet_epoch90.pt"

DEFAULT_MODEL = str(STAGED_MODEL_DIRNAME) + MODEL_FILE
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
FAVICON = APP_DIR / "ui/1f95e.png"  # path to a small image for display in browser tab and social media

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(
        predictor.run,
    )
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=args.share,  # should we create a (temporary) public link on https://gradio.app?
        favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def make_frontend(
    fn: Callable[[Image], str],
):
    """Create a gradio.Interface frontend for an image to segmented mask function."""
    examples_dir = Path(__file__).resolve().parent /"data" / "test_img_subset"
    print(examples_dir)
    example_fnames = [f for f in os.listdir(examples_dir) if f.endswith(".jpg")]
    example_paths = [examples_dir / fname for fname in example_fnames]
    examples = [[str(path)] for path in example_paths]

    allow_flagging = "never"
    readme = "This landcover segmentation prototype is based on previous work by" \
             " Srimannarayana Baratam and Georgios Apostolides as a part of their coursework for " \
             "Computer Vision by Deep Learning (CS4245) offered at TU Delft. " \
             "The implementation of the code was done using PyTorch, it uses U-net architecture " \
             "to perform multi-class semantic segmentation. Our contribution has been to debug the model," \
             "train it, integrate the training with W&B and deploy it as an app."
    # build a basic browser interface to a Python function
    frontend = gr.Interface(
        fn=fn,
        outputs=gr.components.Image(type="pil", label="Landcover Prediction"),
        inputs=[gr.components.Image(type="pil", label="Land cover image"),
                gr.Number(value=0.3, interactive=True)],
        title="📝FSDL 2022 Group Project: Landcover Segmentation",  # what should we display at the top of the page?
        thumbnail=FAVICON,  # what should we display when the link is shared, e.g. on social media?
        description="A prototype landcover segmentation app.",  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=examples,  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
    )

    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = LandcoverSegmentationClassifier(model_path=DEFAULT_MODEL)
            self._predict = model.predict_img

    def run(self, image, scale):
        pred, metrics = self._predict_with_metrics(image, scale)
        self._log_inference(pred, metrics)
        return pred

    def _predict_with_metrics(self, image, scale):
        pred = self._predict(image, scale_factor=scale)

        stats = ImageStat.Stat(image)
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "pred_length": len(pred),
        }
        print(type(pred[0]))
        seg_img = PILIMAGE.fromarray(pred[0])
        return seg_img, metrics

    def _predict_from_endpoint(self, image, scale):
        """Send an image to an endpoint that accepts JSON and return the predicted text.

        The endpoint should expect a base64 representation of the image, encoded as a string,
        under the key "image". It should return the predicted text under the key "pred".

        Parameters
        ----------
        image
            A PIL image of handwritten text to be converted into a string.

        Returns
        -------
        pred
            A string containing the predictor's guess of the text in the image.
        """
        encoded_image = utils.encode_b64_image(image)

        headers = {"Content-type": "application/json"}
        payload = json.dumps({"image": "data:image/png;base64," + encoded_image,
                              "scale_factor": scale})

        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["pred"]

        return pred

    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. "
             "Data is base64-encoded, converted to a utf-8 string, and then set "
             "via a POST request as JSON with the key 'image'. Default is None, "
             "which instead sends the data to a model running locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--share",
        default=False,
        type=bool,
        help=f"share the app externally with others?",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
