# From Denoising Diffusions to Denoising Markov Models

## Running experiments
For the g-and-k distribution example, run
`python main_score_sde.py experiment=conditional dataset.num_quantiles=250 dataset.num_samples=250`

For the MNIST inpainting example, go into the `discrete_ctmc` directory and run 
`python dist_train.py conditional_mnist`

For the ImageNet super-resolution example, go into the `discrete_ctmc` directory and run 
`python train.py conditional_imagenet`

For the SO3 example, run 
`python main.py experiment=so3 dataset.K=16`

For the pose estimation example, run 
`python main.py experiment=symsol`

## References
This codebase is largely based on the existing works of [[1]](https://github.com/oxcsml/riemannian-score-sde) and [[2]](https://github.com/andrew-cr/tauLDR), both also developed at Oxford in the Department of Statistics. It also uses a modified version of [geomstats](https://github.com/oxcsml/geomstats.git) and [haikumodels](https://github.com/abarcel/haikumodels).

[1] Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, Arnaud Doucet, Riemannian Score-Based Generative Modeling, NeurIPS 2022.

[2] Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, Arnaud Doucet, A Continuous Time Framework for Discrete Denoising Models