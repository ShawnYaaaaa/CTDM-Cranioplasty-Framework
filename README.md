# CTDM-Cranioplasty-Framework
Cranioplasty Transformer Diffusion Model (CTDM) — A multimodal clinical system based on Transformer and Diffusion Models for generating cranial defect reconstruction point clouds and diagnostic text.

---

## Overview
With the rapid success of the **Transformer** framework, multimodal learning has become a critical technology driving intelligent medical imaging analysis and clinical decision support. Cranioplasty, a core neurosurgical procedure, requires highly accurate planning and implant design for cranial reconstruction. However, existing approaches face several challenges in **point cloud reconstruction** and **clinical text generation**, such as semantic misalignment, geometric deficiencies, and suboptimal multimodal representation learning.  

To address these challenges, we propose the **Cranioplasty Transformer Diffusion Model (CTDM)**, an innovative framework integrating Transformer-based architectures and Diffusion Models for precise cranial defect reconstruction and clinically relevant diagnostic text generation.

---

## Key Features
- **Multimodal Learning:** Integration of 3D point clouds, medical images, and clinical text.  
- **BLIP (Bootstrapping Language-Image Pre-training):** Improves cross-modal alignment between images and language.  
- **Latent Diffusion Model (LDM):** Enables high-quality visual and structural reconstruction.  
- **PointNet++:** Provides robust point cloud feature extraction and geometry learning.  
- **SNOMED CT Vocabulary:** Guarantees medical domain relevance and standardized terminology.  
- **Custom Loss Functions (PTC/PTM):** Point-Text Contrastive (PTC) and Point-Text Matching (PTM) losses enhance semantic consistency across modalities.  
- **Point-Grounded Text Encoder & Decoder:** Facilitates tight coupling between spatial geometry and clinical linguistic representation.  

---

## Expected Outcomes
- High-precision cranial reconstruction with improved geometric and textural fidelity.  
- Automatic generation of standardized, clinically valuable diagnostic text.  
- Reduced burden on medical professionals with accelerated preoperative planning.  
- Enhanced decision-making accuracy for cranioplasty procedures.  

---

## Suggested Project Structure
