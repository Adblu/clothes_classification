from mmcls.apis import train_model
import time

from src.train_config import CONFIG_PATH
from src.utils.helpers import bring_dataset, prepare_config, prepare_model

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = bring_dataset()

    cfg = prepare_config(CONFIG_PATH)

    model, datasets = prepare_model(cfg)

    print('Model training starts')
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict())
    print('Model finished training')