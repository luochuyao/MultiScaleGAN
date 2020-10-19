 MultiScaleGAN

The source code of paper:
"Experimental Study on Generative Adversarial Network for Precipitation Nowcasting"


This source code involves nine models:
ConvGRU,
ConvLSTM
Multi Scale CNN
ConvGRU GAN,
ConvLSTM GAN,
ConvGRU WGAN,
ConvLSTM WGAN,
Multi Scale GAN,
Multi Scale WGAN,

The first to seventh can be run in the path of experiment/radar/
the last two can be run in the path of multi_scale_gan/avg_runner or wavg_runner/

# About test data
We give the several original test samples(73 images in total) which mention in the paper. It located in the path of data/classic_data/ 


These data need to be preprocessed. The specific steps can be refer to read_files() function in each model file(eg. experiment/radar/convGru.py).
The range of data is 0~255 and the value of 255 presents the default value. Hence, we need to filter it and all pixels whose value larger than 80. 


# About Model
It is difficult to train the model. Therefore, we offer the all models directly this week.


# About maintaining
We will keep maintaining this code recently. 