# ML peak detection for gamma-ray spectrometry

This repository hosts a prototype of my dissertation project, and written for my master's degree in Environmental Sciences with focus on machine learning algorithms at Universidade Estadual Paulista (UNESP).

The algorithm in this project comprises a wavelet transform, a feature extraction routine with Random Forests, and a linear classifier. In tests with artificial spectra, it outperformed classical spectrometry algorithms (such as the Unidentified Second Difference and Library Correlation Nuclide Identification) at every metric.

For more details about the project, please read [my dissertation on the UNESP repository](https://repositorio.unesp.br/handle/11449/148825) (in Portuguese).

## Requirements

- scipy 1.6.1+
- numpy 1.20.1+
- matplotlib 3.3.4+
- scikit-learn 0.24.1+
