# Tensorflow-Keras

<br>安裝 Nvidia 顯卡驅動<br>
每張顯卡版本不同，這邊是1050<br>
設置環境變數<br>
nvcc -V<br>
V10.1.243<br>
這邊要注意如果裝CUDA 10以上 tensorflow至少要>=2.1.0，不然執行tensorflow會error<br>

nvidia-smi<br>
會列出顯卡資訊<br>

pip install tensorflow<br>
pip install keras <br>
conda install cudnn=7.6.5<br>

open jupyter-notebook<br>

from tensorflow.python.client import device_lib<br>

device_lib.list_local_devices()<br>
列出所有cpu與gpu<br>

tensorflow2.0以下就不列了網路上到處都有<br>
而2.0應該tensorflow刪除了Session與API細微的改變<br>
https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session<br>
tensorflow2.0以上代碼<br>

import tensorflow as tf<br>
import os<br>
msg = tf.constant('Hello, TensorFlow!')<br>
tf.print(msg)<br>

os.environ["CUDA_VISIBLE_DEVICES"]="0"<br>
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)<br>


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(<br>
    gpu_options = gpu_options,<br>
    log_device_placement=True))<br><br>
    
然後設置keras GPU進行traing後打開cmd>nvidia-smi

就能看到Processes再跑了<br>

# Keras load model Error
AttributeError: module 'tensorflow' has no attribute 'placeholder'

重新安裝
pip uninstall tensorflow tensorboard tensorboard-plugin-wit tensorflow-estimator keras tensorflow-gpu tensorflow-gpu-estimator-2.2.0

pip install tensorflow tensorboard tensorboard-plugin-wit tensorflow-estimator keras tensorflow-gpu tensorflow-gpu-estimator-2.2.0
