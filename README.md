


# Real-Time Image Based Driver Drowsiness Detection System
Contributors: Harish Yelchur, Subahyn Basu, Evan Wu 


## Problem Statement

Drowsy driving has led to several vehicle accidents over the course of centuries, ever since cars were invented. Road traffic accidents claim approximately 1.35 million lives annually, with 90% of these incidents attributable to driver-related factors such as carelessness and physical impairments, including drowsy driving [5].  Drowsy driving is sometimes overlooked because it is not as serious as driving under the influence of drugs or alcohol, and is often compared in severity with driving under the influence. Drowsy driving can easily impair the driver’s judgement, and losing attention for even a few seconds can be hazardous. 

## Abstract

Develop a deployable image-based driver drowsiness detection (DDD) system by analyzing eye, mouth, and head movements using machine learning (ML) and computer vision (CV). This project will utilize a camera to capture the driver’s face in real-time and employ ML and CV to assess certain behavioral cues that have been proven to be effective in drowsiness detection. 

These may include EAR (Eye Aspect Ratio) to determine if eyes are open or closed, PERCLOS (Percentage of Eyelid Closure), yawn frequency/duration computed over time using MAR (mouth aspect ratio), which is the ratio of the mouth opening, and head droop by calculating the up/down pitch angle.

This project aims to provide a low-cost yet high-accuracy embedded solution, suitable for use in vehicles to enhance road safety.

## Expected Outcomes

- Detect driver drowsiness quickly in real-time and instantly alert the driver
- Understanding the effectiveness of certain drowsiness cues vs others
- Exploring and implementing various machine learning/computer vision techniques for accurate detection and interpretation of facial behavioral cues based on existing literature
- Design, development, and test a complete embedded system
- Ensuring system performance while trying to minimize cost and area

## References

[1] Albadawi, Y., Takruri, M., & Awad, M. (2022). A Review of Recent Developments in Driver Drowsiness Detection Systems. Sensors, 22(5), 2069. https://doi.org/10.3390/s22052069
https://www.mdpi.com/1424-8220/22/5/2069

[2] R. Chinthalachervu, et al., “Driver Drowsiness Detection and Monitoring System using Machine Learning,” J. Phys.: Conf. Ser., vol. 2325, p. 012057, 2022. 
https://iopscience.iop.org/article/10.1088/1742-6596/2325/1/012057/pdf

[3] Albadawi, Y.; AlRedhaei, A.; Takruri, M. Real-Time Machine Learning-Based Driver Drowsiness Detection Using Visual Features. J. Imaging 2023, 9, 91. https://doi.org/10.3390/jimaging9050091
https://www.mdpi.com/2313-433X/9/5/91

[4] Essahraui S, Lamaakal I, El Hamly I, Maleh Y, Ouahbi I, El Makkaoui K, Filali Bouami M, Pławiak P, Alfarraj O, Abd El-Latif AA. Real-Time Driver Drowsiness Detection Using Facial Analysis and Machine Learning Techniques. Sensors (Basel). 2025 Jan 29;25(3):812. doi: 10.3390/s25030812. PMID: 39943451; PMCID: PMC11819803.
https://pmc.ncbi.nlm.nih.gov/articles/PMC11819803/ - GOOD ONE

[5] K. Von C. Golosinda, J. Van D. Marcellones, L. J. P. Ampongol, N. P. Magloyuan and P. D. Cerna, "DriSafePh: An IoT Based Realtime Driver Drowsiness Detection System using Hybrid Machine learning Algorithm," 2024 14th International Conference on Software Technology and Engineering (ICSTE), Macau, China, 2024, pp. 258-265, doi: 10.1109/ICSTE63875.2024.00051.
https://ieeexplore.ieee.org/document/10840392

[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC10650052/

[7] https://ieeexplore.ieee.org/abstract/document/10965037/authors#authors - Real Time DDD using CV and enhanced with ML


# Prerequisites 
- opencv-python 4.12.0
- numpy 2.2.6
- dlib  20.0.0
- imutils 0.5.4
  
## Optional 
- scipy
- scikit-image

