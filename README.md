Code for the method VGPCR, which uses variational inference to approximate the posterior in GP-based crowdsourcing models.
The approach was later extended to deal with larger datasets through the use of Fourier features: https://github.com/pablomorales92/RFFVFF-GPCR 

Full reference:\
Ruiz P., Morales-Álvarez P., Molina R., Katsaggelos A.K.\
Learning from crowds with variational Gaussian processes\
Pattern Recognition, 2019\
DOI: https://doi.org/10.1016/j.patcog.2018.11.021

## Abstract
Solving a supervised learning problem requires to label a training set. This task is traditionally performed by an expert, who provides a label for each sample. The proliferation of social web services (e.g., Amazon Mechanical Turk) has introduced an alternative crowdsourcing approach. Anybody with a computer can register in one of these services and label, either partially or completely, a dataset. The effort of labeling is then shared between a great number of annotators. However, this approach introduces scientifically challenging problems such as combining the unknown expertise of the annotators, handling disagreements on the annotated samples, or detecting the existence of spammer and adversarial annotators. All these problems require probabilistic sound solutions which go beyond the naive use of majority voting plus classical classification methods. In this work we introduce a new crowdsourcing model and inference procedure which trains a Gaussian Process classifier using the noisy labels provided by the annotators. Variational Bayes inference is used to estimate all unknowns. The proposed model can predict the class of new samples and assess the expertise of the involved annotators. Moreover, the Bayesian treatment allows for a solid uncertainty quantification. Since when predicting the class of a new sample we might have access to some annotations for it, we also show how our method can naturally incorporate this additional information. A comprehensive experimental section evaluates the proposed method with synthetic and real experiments, showing that it consistently outperforms other state-of-the-art crowdsourcing approaches.

## Citation
@article{ruiz2019learning,\
  title={Learning from crowds with variational Gaussian processes},\
  author={Ruiz, Pablo and Morales-{\\'A}lvarez, Pablo and Molina, Rafael and Katsaggelos, Aggelos K},\
  journal={Pattern Recognition},\
  volume={88},\
  pages={298--311},\
  year={2019},\
  publisher={Elsevier}\
}
