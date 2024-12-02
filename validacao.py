import multiprocessing
from ultralytics import YOLO

from multiprocessing import freeze_support

validacao = 'validacao.yaml'
hyp = 'hyp.yaml'
model = YOLO('runs/detect/backup/DetectModel_Broca_RecallFocus/weights/best.pt')

if __name__ == '__main__':
    metrics = model.val(
        data=validacao,
        imgsz=1280,
        batch=2, conf=0.6
    )
    for key, value in metrics.results_dict.items():
        print(f'{key}: {value}')

    freeze_support()
