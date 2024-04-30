Code repository for CVPR2024 paper 《Pre-trained Model Guided Fine-Tuning for Zero-Shot Adversarial Robustness》
# Pre-trained-Model-Guided-Fine-Tuning-for-Zero-Shot-Adversarial-Robustness (CVPR 2024)

<p align="center">
  <p align="center" margin-bottom="0px">
    <strong>Sibo Wang</strong></a>
    ·
    <strong>Jie Zhang</strong></a>
    ·
    <strong>Zheng Yuan</strong></a>
    ·
    <strong>Shiguang Shan</strong></a>
    ·
    <p align="center" margin-top="0px"><a href="https://arxiv.org/abs/2401.04350">https://arxiv.org/abs/2401.04350</a></p>
</p>
Large-scale pre-trained vision-language models like CLIP have demonstrated impressive performance across various tasks, and exhibit remarkable zero-shot generalization capability, while they are also vulnerable to imperceptible adversarial examples. 
Existing works typically employ adversarial training (fine-tuning) as a defense method against adversarial examples. 
However, direct application to the CLIP model may result in overfitting, compromising the model's capacity for generalization.
In this paper, we propose Pre-trained Model Guided Adversarial Fine-Tuning (PMG-AFT) method, which leverages supervision from the original pre-trained model by carefully designing an auxiliary branch, to enhance the model's zero-shot adversarial robustness.
Specifically, PMG-AFT minimizes the distance between the features of adversarial examples in the target model and those in the pre-trained model, aiming to preserve the generalization features already captured by the pre-trained model.
Extensive Experiments on 15 zero-shot datasets demonstrate that PMG-AFT significantly outperforms the state-of-the-art method, improving the top-1 robust accuracy by an average of 4.99%.
Furthermore, our approach consistently improves clean accuracy by an average of 8.72%.

### Replace
Replace the files in the replace folder to the source code in your environmet:  

replace `anaconda3/envs/zsrobust/lib/python3.9/site-packages/clip/clip.py` and `anaconda3/envs/zsrobust/lib/python3.9/site-packages/clip/model.py` with clip.py and model.py in the replace folder respectively. 

replace the `anaconda3/envs/zsrobust/lib/python3.9/site-packages/torchvision/datasets` with the files in `replace/torchvision.datasets` 
for updated dataset loader

### Run
python PMG_AFT.py --batch_size 256 --root ./data --dataset tinyImageNet --name wangsibo --train_eps 1 --train_numsteps 2 --train_stepsize 1
