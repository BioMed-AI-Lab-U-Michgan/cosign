from evaluations.evaluator import open_npz_array
import numpy as np
import PIL.Image as image
import os

out_dir = os.environ['DATA_DIR']
filename = os.environ['OPENAI_LOGDIR'] + "/samples_300x256x256x1.npz"
if not os.path.exists(filename):
    filename = os.environ['OPENAI_LOGDIR'] + "/samples_300x256x256x3.npz"

with open_npz_array(filename, "generated") as reader:
    batch = reader.read_batch(300)
    for i, img in enumerate(batch):
        img = img.astype(np.uint8)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        image.fromarray(img).save(os.path.join(out_dir, f"bedroom_{i:07d}.png"))
