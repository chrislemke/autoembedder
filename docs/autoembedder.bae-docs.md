## The model
What the BAE and the vanilla autoencoder have in common is the model's architecture. Usually, when implementing the BAE no changes needs to be made. That is since the BAE handles the autoencoder as a weak learner by itself.

## The learning algorithm and its parts
The idea of boosting in the sense of a Boosting autoencoder ensemble is as based on having an altered learning algorithm. Besides the classical parts of the learning algorithm, for example, backpropagation or calculating the loss, some additional steps need to be implemented. Through the learning process - in each iteration ($i$), instances of the sampled dataset get created using a probability function $\mathbb{P}^{(i)}$. The function is based on the reconstruction error from original dataset ($X^{(0)}$) of the previous iteration. The dataset $X^{(i)}$ should contain fewer outliers than the $X^{(i-1)}$. This is going to be archived by assigning weights to the datasets entries, so that entries with a higher reconstruction error have a lower probability of being sampled.

Probability function:<br>
$\mathbb{P}^{(i+1)}_{x} = \frac{1/e^{(i)}_x}{\sum{1/e^{(i)}_x}}$

Reconstruction error:<br>
$e^{(i)}_x = (\left\lVert x-AE^{(i)}(x) \right\rVert_2)^2$<br>
The difference to the reconstruction error of the vanilla AE is the iterations ($i$) factor as well as the L2-norm within this function.

According to the $\mathbb{P}^{(i)}$ $X^{(i+1)}$ contains less outliers than $X^{(i)}$ and therefore the learning process concentrates the inliers and the outlier get lower impact on the model.

Savari et al. define one advantage of the BAE with the capability for a weighted consensus. The different components archive though the iterations an "agreement" between the weak learners. It is important to mention that - to follow the idea of boosting - several autoencoders are used improve the performance achieved by a single autoencoder (the first iteration).

Since the BAE is about boosting, the algorithm is split into two parts: the training phase and the consensus phase. In the boosting phase, the AE is trained with $X^{(i)}$ - the sampled view of $X^{(0)}$. This step is not different from the training of the vanilla AE. After this we proceed by sampling $X^{(i+1)}$ with the entries whose reconstruction error is lower. The sampling process used $X^{(0)}$ and the probability function $\mathbb{P}$ to create a new dataset. This training phase is repeated for all components of the BAE ensemble.<br>
In the consensus phase we use the the following function to calculate the weights:

$w^{(i)}=\frac{1/\sum\limits_{x\in X} e^{(i)}_x}{\sum\limits_{i=1}^{m-1}(1/\sum\limits_{x\in X^{(i)}}e^{i}_x)}$<br>

Those weights are used to calculate the outlier score for each $x$:

$\bar e_x=\sum\limits^{m-1}_{i=1}w^{(i)}e_x^{(i)}$<br>

$\bar e_x$ is than assigned to each $x \in X^{(0)}$.
