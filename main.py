from yolov5.detect import run
from yolov5.model import load_model

model = load_model(weights=r'C:\Users\estagio.sst17\source\repos\AnnotateImages\NeuralNetworkModels\yolov5x.pt')

result = run(source=r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\Banco', model=model, save_txt=True, conf_thres=0.65)

