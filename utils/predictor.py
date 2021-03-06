import numpy as np
import joblib
from utils.batch_manager import fit_into_batch, extract_from_batch
from utils.preprocessstock import preprocess

class Predictor():
    def __init__(self, model_type:str, batch_size:int = 32):
        self.model_path = {
                'tdnn': 'weights/tdnn.h5',
                'tdnn_pso': 'weights/tdnnpso.h5',
                'tdnn_de': 'weights/tdnnde.h5',
                'rf': 'weights/rf.sav',
                'svm': 'weights/svm.sav'
            }
        self.batch_size = batch_size
        self.model_type = model_type
        self.x = None
        self.y = None
        self.max = None
        self.min = None
        if self.model_type.lower() not in self.model_path.keys():
            raise Exception('Unsupported model type.')
        if 'tdnn' in self.model_type.lower():
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path[self.model_type])
        elif self.model_type.lower() == 'rf':
            self.model = joblib.load(self.model_path[self.model_type])
        elif self.model_type.lower() == 'svm':
            self.model = joblib.load(self.model_path[self.model_type])

    def load_transform(self, stock:str):
        features, labels, getmax, getmin = preprocess(stock)
        self.x, self.y, self.max, self.min = features.astype('float32'), labels.astype('float32'), getmax, getmin
        self.x = np.expand_dims(self.x, axis=1)
        self.x = np.expand_dims(self.x, axis=-1)
        return self.x, self.y, self.max, self.min

    def predict(self, data=None):
        if data is None:
            if self.x is None:
                raise Exception('input data required.')
            else:
                data = self.x
        pred = None
        try:
            if 'tdnn' in self.model_type:
                data, og_n_samples = fit_into_batch(data, self.batch_size)
                pred = self.model.predict(data)
                pred = extract_from_batch(pred, og_n_samples)
            else:
                data = data.reshape(data.shape[0], 35).astype('float32')
                pred = self.model.predict(data)
        except Exception as e:
            print(f'Error while prediction {e}')
        return pred
