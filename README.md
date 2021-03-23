# ML peak detection on gamma-ray spectra

This repository hosts a prototype of my dissertation project, which I carried out for my master's degree in Environmental Sciences with focus on machine learning algorithms at Universidade Estadual Paulista (UNESP).

The algorithm in this project comprises a wavelet transform, a feature extraction routine with Random Forests, and a linear classifier. In tests with artificial spectra, it outperformed classical spectrometry algorithms (such as the Unidentified Second Difference and Library Correlation Nuclide Identification).

For more details about the project, please read [my dissertation on the UNESP repository](https://repositorio.unesp.br/handle/11449/148825).

# Required modules

- sklearn
- scipy
- numpy
- matplotlib
- pywavelet