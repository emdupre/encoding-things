from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm.contrib.concurrent import thread_map
from transformers import AutoImageProcessor, AutoModel


@click.command()
@click.option(
    "--subj_id",
    type=click.Choice(["subj01", "subj02", "subj05", "subj07"]),
    default="subj01",
    help="Subject name.",
)
@click.option(
    "--stimuli_dir",
    default="/home/emdupre/links/projects/def-pbellec/emdupre/NSD/stimuli",
    help="Stimulus directory.",
)
@click.option(
    "--output_dir",
    default="/scratch/emdupre/NSD/stimuli.algonauts-predictions",
    help="Output_directory",
)
def main(subj_id, stimuli_dir, output_dir):
    stimuli_dir = Path(stimuli_dir)
    # "/Users/emdupre/Desktop/UdeM-Projects/things-encode/cneuromod.things.stimuli"
    # "/Users/emdupre/Desktop/UdeM-Projects/nsd/stimuli"
    # output_dir = Path(stimuli_dir.parent, "stimuli.algonauts-predictions")
    output_dir.mkdir(exist_ok=True, parents=True)

    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-large",
        revision="main",
        local_files_only=True,
    )
    # whole brain model, 327684 vertices in fsaverage
    model = AutoModel.from_pretrained(
        "huzey/brain-synthesizer",
        revision="274fbb83069b7867f6e044052db7732da56679e7",
        local_files_only=True,
        trust_remote_code=True,
        subject=subj_id,  # options: subj01, subj02, subj05, subj07
    )

    def get_prediction(img_path):
        image = Image.open(img_path).convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        prediction = model(pixel_values).detach().numpy()
        # torch.Size([1, 327684]), vertices in fsaverage,
        # first half is left hemisphere, second half right hemisphere
        # print(prediction.shape)

        # Save prediction to .npy file
        np.save(
            Path(
                output_dir,
                f"brain-synthesizer-{subj_id}-{img_path.stem}-prediction.npy",
            ),
            prediction,
        )
        return

    img_paths = list(Path(stimuli_dir).rglob("*.jpg"))
    thread_map(get_prediction, img_paths, max_workers=50)

    return


if __name__ == "__main__":
    main()
