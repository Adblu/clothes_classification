import mmcv
from glob import glob
from src.mmclassification.mmcls.apis import inference_model
from src.train_config import CONFIG_PATH
from src.utils.helpers import prepare_config, prepare_model
import random
from PIL import Image

if __name__ == "__main__":
    test_files = glob('/home/n/Documents/STER/src/data/clothes/test/*/*.*')

    cfg = prepare_config(CONFIG_PATH)
    model, datasets = prepare_model(cfg)
    model.cfg = cfg

    for i in range(10):

        random_idx = random.randrange(1, len(test_files), 1)

        img = mmcv.imread(test_files[random_idx])
        result = inference_model(model, img)
        result_img = model.show_result(img, result, show=False)

        Image.fromarray(result_img).save(f"../figures/your_file_{random_idx}.jpeg")









