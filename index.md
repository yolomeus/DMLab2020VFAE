# Data Mining Lab 2020: The Variational Fair Autoencoder Project
 
The goal of this project was to reproduce results from 
"[The Variational Fair Autoencoder](https://arxiv.org/abs/1511.00830)" and to extend it with further ideas.   

The VFAE is a variant of the [Variational Autoencoder](https://arxiv.org/abs/1312.6114) neural network architecture 
with the goal of producing **fair** latent representations. In the fair classification setting, input features contain a 
so called **sensible** or **protected** feature **s** that indicates membership to a protected (e.g. minority-) class. 
This could for example be gender, religion or age.  
The VFAE tries to produce latent representations of the inputs that no longer contain sensible information about **s**
while staying useful for downstream tasks like classification.