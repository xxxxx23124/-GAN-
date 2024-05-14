# -GAN-
佘兴黔的2024本科毕业设计，基于对抗生成网络（GAN）生成艺术图片。由于设备限制，所有生成器和判别器都是基于处理64x64像素的图像设计。
使用pytorch，torch.__version__ == 2.2.2+cu121， torch.version.cuda == 12.1， torch.backends.cudnn.version() == 8801
显卡为笔记本2070s 8g显存，G13和D9比较特殊，其余模型均是在此显卡上运行。
每个生成器模型都有不同的缺陷，生成器随着编号渐渐弥补这些缺陷。
最终采用13_5号模型做为生成器，第9_4号判别器做为判别器
以下是在https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models
数据集上的24个epoch的训练结果
![24-300](https://github.com/xxxxx23124/-GAN-/assets/137014884/a7de3f9b-6087-4b8d-baa5-6ff98f190be3)
以下是训练47epoch的视频，每张图间隔200个迭代


https://github.com/xxxxx23124/-GAN-/assets/137014884/18e4fb40-80be-41ca-ae1a-8c10368dd895

