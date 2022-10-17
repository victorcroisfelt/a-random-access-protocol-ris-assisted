# A Random Access Protocol for RIS-Aided Wireless Communications
This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the article mentioned below and also encourage and accelerate further research on this topic:

V. Croisfelt, F. Saggese, I. Leyva-Mayorga, R. Kotaba, G. Gradoni and P. Popovski, ["A Random Access Protocol for RIS-Aided Wireless Communications,"](https://ieeexplore.ieee.org/document/9833984) 2022 IEEE 23rd International Workshop on Signal Processing Advances in Wireless Communication (SPAWC), 2022, pp. 1-5, doi: 10.1109/SPAWC51304.2022.9833984.

A pre-print version is available on: https://arxiv.org/abs/2203.03377, which has a different content from the published one.

I hope this content helps in your reaseach and contributes to building the precepts behind open science. Remarkably, in order to boost the idea of open science and further drive the evolution of science, I also motivate you to share your published results to the public.

If you have any questions and if you have encountered any inconsistency, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## Abstract
Reconfigurable intelligent surfaces (RISs) are arrays of passive elements that can control the reflection of the incident electromagnetic waves. While RISs are particularly useful to avoid blockages, the protocol aspects for their implementation have been largely overlooked. In this paper, we devise a random access protocol for a RIS-assisted wireless communication setting. Rather than tailoring RIS reflections to meet the positions of users’ equipment (UEs), our protocol relies on a finite set of RIS configurations designed to cover the area of interest. The protocol is comprised of a downlink training phase followed by an uplink access phase. During these phases, a base station (BS) controls the RIS to sweep through its configurations. The UEs then receive training signals to measure their channel quality for the different RIS configurations and refine their access policies. Numerical results show that our protocol increases the average number of successful access attempts; however, at the expense of increased access delay due to the realization of a training period. Promising results are further observed in scenarios with a high access load.

## Content
The codes provided here can be used to simulate Fig. 2 of the paper. The code is divided in the following way:
  - scripts starting with the keyword "plot_" actually plots the figures using matplotlib.
  - scripts starting with the keyword "data_" are used to generate data for curves that require a lot of processing. The data is saved in the /data folder and used by the respective "plot_" scripts.
  - scripts starting with the keyword "lookup_" are used to exhaustively find parameters, such as: number of nearby APs, Ccal_size, number of pilot-serving APs, Lmax, and effective DL transmit power for Estimator 3, delta. Considering the practical scenario, it also makes use of method proposed in Algorithm 1. 

Further details about each file can be found inside them.

## Citing this Repository and License
This code is subject to the MIT license. If you use any part of this repository for research, please consider to cite our aforementioned work.
