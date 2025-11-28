1. The DFBN joint distribution is implemented as the composition of an Autoregressive Logistic Regression, a Fuzzy Inference Layer, and an FG-LSTM Encoder.
2. The Fuzzy inference layer includes gaze feature extraction, fuzzy clustering + membership function fitting, unsupervised Wang-Mender rule extraction, and Mamdani inference.
3. The processed gaze sequence dataset is available in sequence_new_aug2.csv, which contains detailed time steps, gaze duration, Individual attention values, group attention values, and duration-dependent attention values.
4. The detailed mathematical principles will be published after the paper is published.
5. The original video data will be released after the paper is published, example data is shown in the video.
