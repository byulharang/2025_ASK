# U-Net with RIR conditional Injection for Depth Inpainting
At **Domestic Conference** held by **The Acoustical Society of Korea** (ASK) 

*Conf. 2025년도 한국음향학회 춘계학술발표대회 및 제40회 수중음향학 학술발표회*
<img src="https://github.com/byulharang/2025_ASK/blob/main/images/%ED%95%99%ED%9A%8C%20%ED%91%9C%EC%A7%80.png" alt="conference poster" width="500" />

---

**Research Period:** 2025 Winter-April (*during URP program*) <br>
**Advisor:** Jung-Woo Choi (KAIST EE) <br>
**Conference Extended Abstract:** 🧾 [PDF](https://drive.google.com/file/d/1AHi8pkvUU6dJaGWIJIIBKJAqtirqbtON/view?usp=sharing) <br>
**Conference Presentation:** 🔬 [Google Slide](https://docs.google.com/presentation/d/1HhTRdiQzxTSDWnr006PSXB0aY1t0WpsM3ibcu-Sb260/edit?usp=sharing) <br>
**Data & Experiment Logs:** 🌐 [Notion](https://kiwi-primrose-e33.notion.site/URP-16d30761238f8068aec6f9576ef4bee2?source=copy_link) *Identical links in URP repo*

---

# Abstract
﻿Reconstructing the indoor structures is crucial for augmented/extended reality applications or physical interactions. 
The structural information can be obtained from indoor depth panoramas; 
however, acquiring a complete depth panorama under typical conditions remains challenging due to the limited field of view. 
We propose the CNN based conditional RIR injection model that reconstruct the full depth panorama from the partial panorama and room impulse response. 
The model outperform the same model, but without the RIR condition in evaluation metrics and perceptual comparison.

<img src="https://github.com/byulharang/2025_ASK/blob/main/images/architecture.png" alt="Model Flow" width="900" />

# Result 
Our proposed model outperform the Vision only model without the RIR injection in
* **Peak-signal-to-noise (PSNR):** High value refer better image quality
* **Structural similiarity index map (SSIM):** High value refer better structural, luminance, contrast quality

<img src="https://github.com/byulharang/2025_ASK/blob/main/images/metric.png" alt="PSNR and SSIM" width = "800" />

* **Perceptual Quality with naive eyes:** Proposed model estimate the existence of structure like wall

<img src="https://github.com/byulharang/2025_ASK/blob/main/images/perceptual%20Q.png" alt="Perceptual Result Comparison" width="900" />

# Future Work
Dealt in the URP repo. since the proposed model in **HERE** considered as intermediate model on the URP Program.


# Acknowledge
This study was supported by the Undergraduate Research Participation (URP) program funded by the Korea Advanced Institute of Science and Technology (KAIST).
