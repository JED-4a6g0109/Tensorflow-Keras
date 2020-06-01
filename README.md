# Tensorflow-Keras

<br>安裝 Nvidia 顯卡驅動<br>
每張顯卡版本不同，這邊是1050<br>
設置環境變數<br>
nvcc -V<br>
V10.1.243<br>
這邊要注意如果裝CUDA 10以上 tensorflow至少要>=2.1.0，不然執行tensorflow會error<br>

nvidia-smi<br>

+-----------------------------------------------------------------------------+<br>
| NVIDIA-SMI 441.45       Driver Version: 441.45       CUDA Version: 10.2     |<br>
|-------------------------------+----------------------+----------------------+<br>
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |<br>
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |<br>
|===============================+======================+======================|<br>
|   0  GeForce GTX 1050   WDDM  | 00000000:01:00.0 Off |                  N/A |<br>
| N/A   53C    P8    N/A /  N/A |   3386MiB /  4096MiB |      0%      Default |<br>
+-------------------------------+----------------------+----------------------+<br>

+-----------------------------------------------------------------------------+<br>
| Processes:                                                       GPU Memory |<br>
|  GPU       PID   Type   Process name                             Usage      |<br>
|=============================================================================|<br>
|                                                                             |<br>
+-----------------------------------------------------------------------------+<br>


pip install tensorflow
pip install keras 
conda install cudnn=7.6.5

open jupyter-notebook

from tensorflow.python.client import device_lib

device_lib.list_local_devices()
列出所有cpu與gpu

tensorflow2.0以下就不列了網路上到處都有
而2.0應該tensorflow刪除了Session與API細微的改變
https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session
tensorflow2.0以上代碼

import tensorflow as tf
import os
msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    gpu_options = gpu_options,
    log_device_placement=True))
    
然後設置keras GPU進行traing後打開cmd>nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 441.45       Driver Version: 441.45       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1050   WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   51C    P8    N/A /  N/A |   3386MiB /  4096MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      5020      C   D:\anaconda3\python.exe                    N/A      |
+-----------------------------------------------------------------------------+

就能看到Processes再跑了
