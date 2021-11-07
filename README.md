# Neuronal-Network-Classifier
Neuronal Network Classifier implementation and comparisons using PyTorch, Keras, Tensorflow and numpy from scratch
```sh
conda create -n categorical python=3.9
conda activate categorical
conda list env

conda install numpy
conda install pandas
conda install scikit-learn

# windows 10/11 cuda 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# windows 10/11 cpu
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# mac os x
conda install pytorch torchvision torchaudio -c pytorch

conda install -c conda-forge keras
conda install tensorflow

# install all at once
conda install numpy pandas scikit-learn tensorflow pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge keras
```