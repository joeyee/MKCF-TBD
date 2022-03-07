# Multiple Kernelized Correlation Filters based Track-Before-Detect Algorithm for Tracking Weak and Extended Target in Marine Radar Systems

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)   
This is the **python** implementation of the - 
[Multiple Kernelized Correlation Filters based Track-Before-Detect Algorithm for Tracking Weak and Extended Target in Marine Radar Systems](https://ieeexplore.ieee.org/document/9709567).

<!---[Source paper](https://) of the preprint version.--->

## Introduction
We have extended [MKCF](https://github.com/joeyee/MKCF/) to track weak target in 
the framework of multi-frame track-before-detect (MF-TBD). 
Now the source paper is under the second review. 

Simulated Rayleigh and K distributed sea clutter with varied
PSNR (peak signal to noise ratio) are given in the file []().


## Requirements
- python - 3.9.1
- scipy  - 1.6.0
- opencv-python - 4.5.1
- PIL    - 8.1.0

## How to use the code

### step 1 


File 'MCF_TBD_xxx.py' implements the algorithm of MKCF-TBD.

File 'DP_TBD_Grossi_ETTsim_xxx.py' implements MSAR-TBD mentioned in the paper.

File 'DP_TBD_LELR_ETTsim_xxx.py' implements WTSA-TBD mentioned in the paper.

File 'cfar_segmentation_xxx.py' implements the CFAR and Segmentation pre-processing.

File 'MCF_GROSS_LELR_Simulation_rayleigh_xxx.py' runs the three trackers in Rayleigh distributed sea clutter.

File 'MCF_GROSS_LELR_Simulation_K_xxx.py' runs the three trackers in K distributed sea clutter.

Run the last two files will see the simulation results.

If you find these codes are useful to your research, please cite our MKCF-TBD paper [[Zhou et al., 2022]](https://ieeexplore.ieee.org/document/9709567).
## Reference:

[[Zhou et al., 2022]](https://ieeexplore.ieee.org/document/9709567)
Yi Zhou, Hang Su, Tian Shuai, Xiaoming Liu, Jidong Suo, 
Multiple Kernelized Correlation Filters based Track-Before-Detect Algorithm for Tracking Weak and Extended Target in Marine Radar Systems,
 IEEE Transactions on Aerospace and Electronic Systems, 2022.


[[Zhou et al., 2019]](https://ieeexplore.ieee.org/document/8718392)
Yi Zhou, Tian Wang, Ronghua Hu, Hang Su, Yi Liu, 
Xiaoming Liu, Jidong Suo, Hichem Snoussi, 
Multiple Kernelized Correlation Filters (MKCF) for Extended Object 
Tracking Using X-band Marine Radar Data, 
IEEE Transactions on Signal Processing, vol. 67, no. 14, pp. 3676-3688, 2019.

[[Grossi et al., 2013]](https://ieeexplore.ieee.org/document/6475194)
E.Grossi, M.Lops, and L.Venturino, A novel dynamic programming algorithm
  for track-before-detect in radar systems, IEEE Transactions on Signal
  Processing, vol.61, no.10, pp.2608 -- 2619, May 2013.

[[Jiang et al., 2017]](https://ieeexplore.ieee.org/document/7843642) 
Haichao Jiang, Wei Yi, Thia Kirubarajan, Lingjiang Kong, and Xiaobo Yang, 
Multiframe radar detection of fluctuating targets using phase information, 
IEEE Transactions on Aerospace and Electronic Systems, vol.53, no.2, pp.736 --
  749, April 2017.
