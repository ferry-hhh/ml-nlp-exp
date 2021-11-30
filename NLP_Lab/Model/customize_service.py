from model_service.tfserving_model_service import TfServingBaseService
import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_SEQUENCE_LENGTH = 1000

class text_classify_service(TfServingBaseService):

    def _preprocess(self, data):
        tokenizer = joblib.load(os.path.join(self.model_path, 'token_result.pkl'))
        x_pred_temp1 = tokenizer.texts_to_sequences(list(data.values()))
        print('x_pred_temp1=', x_pred_temp1)
        x_pred_temp2 = pad_sequences(x_pred_temp1, maxlen=MAX_SEQUENCE_LENGTH)
        preprocessed_data=tf.convert_to_tensor(x_pred_temp2, dtype=tf.int32)
        print('dim=', preprocessed_data.ndim)
        print('dtype=', preprocessed_data.dtype)
        print('preprocessed_data=', preprocessed_data)
        return preprocessed_data

    def _postprocess(self, data):
        labels_index = {0: 'alt.atheism', 1: 'comp.graphics', 2: 'comp.os.ms-windows.misc', 3: 'comp.sys.ibm.pc.hardware', 4: 'comp.sys.mac.hardware', 5: 'comp.windows.x', 6: 'misc.forsale', 7: 'rec.autos', 8: 'rec.motorcycles', 9: 'rec.sport.baseball', 10: 'rec.sport.hockey', 11: 'sci.crypt', 12: 'sci.electronics', 13: 'sci.med', 14: 'sci.space', 15: 'soc.religion.christian', 16: 'talk.politics.guns', 17: 'talk.politics.mideast', 18: 'talk.politics.misc', 19: 'talk.religion.misc'}
        print('pred_res=', data)
        data_value = np.array(list(data.values()))
        infer_output = labels_index[data_value.argmax()]
        return infer_output
