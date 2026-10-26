# Supported distributions

[Documentation index](README.md) · [Main README](../README.md) ·
[Family examples](../examples/IMPUTATION.md)

| `family` | Additional constructor argument | Estimated Parameters | Fixed Parameters |
|---|---|---|---|
| `"gaussian"` | — | mean and standard deviation | |
| `"gaussian_known_sigma"` | `sigma=...` | mean | standard deviation (`sigma`) |
| `"poisson"` | — | rate | |
| `"bernoulli"` | — | success probability | |
| `"exponential"` | — | mean | |
| `"laplace_known_mu"` | `mu=...` | scale | location (`mu`) |
| `"chi_square"` | — | degrees of freedom | |
| `"multinomial"` | ordered `categories=[...]` | all category probabilities | |

## Exponential family representation

EFSC uses the canonical exponential-family form

$$\log p(y\mid\boldsymbol\eta)=c(y)+\boldsymbol\eta^\top\mathbf t(y)-a(\boldsymbol\eta),$$

where $\mathbf t(y)$ is the sufficient statistic, $\boldsymbol\eta$ is the
natural parameter, $a(\boldsymbol\eta)$ is the log-partition function, and
$c(y)$ is the carrier term. See the corresponding
[Wikipedia entry](https://en.wikipedia.org/wiki/Exponential_family) for a
friendly introduction and details on additional families. For more rigorous
treatments, see Efron's
[*Exponential Families in Theory and Practice*](https://www.cambridge.org/core/books/exponential-families-in-theory-and-practice/45CE4C98DEAB5EC9D6934E9DAC002F5F)
or Bickel and Doksum's
[*Mathematical Statistics: Basic Ideas and Selected Topics, Volume I*](https://www.routledge.com/Mathematical-Statistics-Basic-Ideas-and-Selected-Topics-Volume-I-Second-Edition/Bickel-Doksum/p/book/9781498723800).

## Gaussian

$$p(y\mid\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right).$$

Support: $y\in\mathbb R$.

Parameters: mean $\mu\in\mathbb R$ and standard deviation $\sigma>0$.

Exponential-family representation:

$$\mathbf t(y)=(y,y^2)^\top,\qquad \boldsymbol\eta(\mu,\sigma)=\left(\frac{\mu}{\sigma^2},-\frac{1}{2\sigma^2}\right)^\top,$$

$$a(\boldsymbol\eta)=-\frac{\eta_1^2}{4\eta_2}-\frac{1}{2}\log(-2\eta_2),\qquad c(y)=-\frac{1}{2}\log(2\pi).$$

## Poisson

$$p(y\mid\lambda)=\frac{\lambda^y e^{-\lambda}}{y!}.$$

Support: $y\in\{0,1,2,\ldots\}$.

Parameter: rate $\lambda>0$.

Exponential-family representation:

$$t(y)=y,\qquad \eta(\lambda)=\log\lambda,\qquad a(\eta)=\exp(\eta),\qquad c(y)=-\log(y!).$$

## Bernoulli

$$p(y\mid\pi)=\pi^y(1-\pi)^{1-y}.$$

Support: $y\in\{0,1\}$.

Parameter: success probability $0<\pi<1$.

Exponential-family representation:

$$t(y)=y,\qquad \eta(\pi)=\log\left(\frac{\pi}{1-\pi}\right),\qquad a(\eta)=\log(1+\exp(\eta)),\qquad c(y)=0.$$

## Exponential

$$p(y\mid\lambda)=\lambda e^{-\lambda y}.$$

Support: $y\geq0$.

Parameter: rate $\lambda>0$, with mean $1/\lambda$.

Exponential-family representation:

$$t(y)=y,\qquad \eta(\lambda)=-\lambda,\qquad a(\eta)=-\log(-\eta),\qquad c(y)=0.$$

## Laplace with known location

$$p(y\mid\mu,b)=\frac{1}{2b}\exp\left(-\frac{|y-\mu|}{b}\right).$$

Support: $y\in\mathbb R$.

Parameters: scale $b>0$; the location $\mu\in\mathbb R$ is fixed by the
user.

Exponential-family representation:

$$t(y)=|y-\mu|,\qquad \eta(b)=-\frac{1}{b},\qquad a(\eta)=\log\left(-\frac{2}{\eta}\right),\qquad c(y)=0.$$

## Chi-square

$$p(y\mid\nu)=\frac{y^{\nu/2-1}e^{-y/2}}{2^{\nu/2}\Gamma(\nu/2)}.$$

Support: $y>0$.

Parameter: degrees of freedom $\nu>0$.

Exponential-family representation:

$$t(y)=\log y,\qquad \eta(\nu)=\frac{\nu}{2}-1,\qquad a(\eta)=\log\Gamma(\eta+1)+(\eta+1)\log 2,\qquad c(y)=-\frac{y}{2}.$$

## Multinomial

$$p(\mathbf y\mid m,\boldsymbol\pi)=\frac{m!}{\prod_{c=1}^{C}y_c!}\prod_{c=1}^{C}\pi_c^{y_c}.$$

Support: $\mathbf y\in\mathbb N_0^C$ with
$\sum_{c=1}^{C}y_c=m$.

Parameters: cell size $m\in\mathbb N$ and category probabilities
$\boldsymbol\pi\in(0,1)^C$ with $\sum_{c=1}^{C}\pi_c=1$.

For multinomial data, the final category is the reference category.

Exponential-family representation:

$$\mathbf t(\mathbf y)=(y_1,\ldots,y_{C-1})^\top,\qquad \eta_k(\boldsymbol\pi)=\log\left(\frac{\pi_k}{\pi_C}\right),\quad k=1,\ldots,C-1,$$

$$a(\boldsymbol\eta)=m\log\left(1+\sum_{k=1}^{C-1}\exp(\eta_k)\right),\qquad c(\mathbf y)=\log(m!)-\sum_{k=1}^{C}\log(y_k!).$$
