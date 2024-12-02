import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
arquivo_config = 'configs_modelo.yaml'

model = YOLO('yolov8l.pt').to(device)

if __name__ == '__main__':
    model.train(
        name='DetectModel_Mariposa_1500-yolov8L-augmentTrue',
        data=arquivo_config,
        epochs=1500,
        imgsz=1024,
        batch=2,
        patience=0,
        augment=True,  # Habilita aumentos de dados padrão
        device=device
    )
    freeze_support()
