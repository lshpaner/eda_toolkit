.. _theoretical_overview:   

.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: ../assets/eda_toolkit_logo.svg
   :alt: EDA Toolkit Logo
   :align: left
   :width: 300px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 100px;"></div>

\


Gaussian Assumption for Normality
----------------------------------

The Gaussian (normal) distribution is a key assumption in many statistical methods. It is mathematically represented by the probability density function (PDF):

.. math::

    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

where:

- :math:`\mu` is the mean
- :math:`\sigma^2` is the variance

In a normally distributed dataset:

- 68% of data falls within :math:`\mu \pm \sigma`
- 95% within :math:`\mu \pm 2\sigma`
- 99.7% within :math:`\mu \pm 3\sigma`

.. raw:: html

   <div class="no-click">

.. image:: ../assets/normal_distribution.png
   :alt: KDE Distributions - KDE (+) Histograms (Density)
   :align: center
   :width: 950px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 50px;"></div>


Histograms and Kernel Density Estimation (KDE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Histograms**:

- Visualize data distribution by binning values and counting frequencies.
- If data is Gaussian, the histogram approximates a bell curve.

**KDE**:

- A non-parametric way to estimate the PDF by smoothing individual data points with a kernel function.
- The KDE for a dataset :math:`X = \{x_1, x_2, \ldots, x_n\}` is given by:

.. math::

    \hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)

where:

- :math:`K` is the kernel function (often Gaussian)
- :math:`h` is the bandwidth (smoothing parameter)

.. raw:: html

   <b><a href="../eda_plots.html#kde_hist_plots">Combined Use of Histograms and KDE</a></b>

   

\

- **Histograms** offer a discrete, binned view of the data.
- **KDE** provides a smooth, continuous estimate of the underlying distribution.
- Together, they effectively illustrate how well the data aligns with the Gaussian assumption, highlighting any deviations from normality.


Pearson Correlation Coefficient
--------------------------------

The Pearson correlation coefficient, often denoted as :math:`r`, is a measure of 
the linear relationship between two variables. It quantifies the degree to which 
a change in one variable is associated with a change in another variable. The 
Pearson correlation ranges from :math:`-1` to :math:`1`, where:

- :math:`r = 1` indicates a perfect positive linear relationship.
- :math:`r = -1` indicates a perfect negative linear relationship.
- :math:`r = 0` indicates no linear relationship.

The Pearson correlation coefficient between two variables :math:`X` and :math:`Y` is defined as:

.. math::

    r_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}

where:

- :math:`\text{Cov}(X, Y)` is the covariance of :math:`X` and :math:`Y`.
- :math:`\sigma_X` is the standard deviation of :math:`X`.
- :math:`\sigma_Y` is the standard deviation of :math:`Y`.

Covariance measures how much two variables change together. It is defined as:

.. math::

    \text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \mu_X)(Y_i - \mu_Y)

where:

- :math:`n` is the number of data points.
- :math:`X_i` and :math:`Y_i` are the individual data points.
- :math:`\mu_X` and :math:`\mu_Y` are the means of :math:`X` and :math:`Y`.

The standard deviation measures the dispersion or spread of a set of values. For 
a variable :math:`X`, the standard deviation :math:`\sigma_X` is:

.. math::

    \sigma_X = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (X_i - \mu_X)^2}

Substituting the covariance and standard deviation into the Pearson correlation formula:

.. math::

    r_{XY} = \frac{\sum_{i=1}^{n} (X_i - \mu_X)(Y_i - \mu_Y)}{\sqrt{\sum_{i=1}^{n} (X_i - \mu_X)^2} \sqrt{\sum_{i=1}^{n} (Y_i - \mu_Y)^2}}

This formula normalizes the covariance by the product of the standard deviations of the two variables, resulting in a dimensionless coefficient that indicates the strength and direction of the linear relationship between :math:`X` and :math:`Y`.

- :math:`r > 0`: Positive correlation. As :math:`X` increases, :math:`Y` tends to increase.
- :math:`r < 0`: Negative correlation. As :math:`X` increases, :math:`Y` tends to decrease.
- :math:`r = 0`: No linear correlation. There is no consistent linear relationship between :math:`X` and :math:`Y`.

The closer the value of :math:`r` is to :math:`\pm 1`, the stronger the linear relationship between the two variables.


Partial Dependence Foundations
--------------------------------

Let :math:`\mathbf{X}` represent the complete set of input features for a machine 
learning model, where :math:`\mathbf{X} = \{X_1, X_2, \dots, X_p\}`. Suppose we're 
particularly interested in a subset of these features, denoted by :math:`\mathbf{X}_S`. 
The complementary set, :math:`\mathbf{X}_C`, contains all the features in :math:`\mathbf{X}` 
that are not in :math:`\mathbf{X}_S`. Mathematically, this relationship is expressed as:

.. math::

   \mathbf{X}_C = \mathbf{X} \setminus \mathbf{X}_S

where :math:`\mathbf{X}_C` is the set of features in :math:`\mathbf{X}` after 
removing the features in :math:`\mathbf{X}_S`.

Partial Dependence Plots (PDPs) are used to illustrate the effect of the features 
in :math:`\mathbf{X}_S` on the model's predictions, while averaging out the 
influence of the features in :math:`\mathbf{X}_C`. This is mathematically defined as:

.. math::
   \begin{align*}
   \text{PD}_{\mathbf{X}_S}(x_S) &= \mathbb{E}_{\mathbf{X}_C} \left[ f(x_S, \mathbf{X}_C) \right] \\
   &= \int f(x_S, x_C) \, p(x_C) \, dx_C \\
   &= \int \left( \int f(x_S, x_C) \, p(x_C \mid x_S) \, dx_C \right) p(x_S) \, dx_S
   \end{align*}


where:

- :math:`\mathbb{E}_{\mathbf{X}_C} \left[ \cdot \right]` indicates that we are taking the expected value over the possible values of the features in the set :math:`\mathbf{X}_C`.
- :math:`p(x_C)` represents the probability density function of the features in :math:`\mathbf{X}_C`.

This operation effectively summarizes the model's output over all potential values of the complementary features, providing a clear view of how the features in :math:`\mathbf{X}_S` alone impact the model's predictions.


**2D Partial Dependence Plots**

Consider a trained machine learning model :ref:`2D Partial Dependence Plots <2D_Partial_Dependence_Plots>`:math:`f(\mathbf{X})`, where :math:`\mathbf{X} = (X_1, X_2, \dots, X_p)` represents the vector of input features. The partial dependence of the predicted response :math:`\hat{y}` on a single feature :math:`X_j` is defined as:

.. math::

   \text{PD}(X_j) = \frac{1}{n} \sum_{i=1}^{n} f(X_j, \mathbf{X}_{C_i})

where:

- :math:`X_j` is the feature of interest.
- :math:`\mathbf{X}_{C_i}` represents the complement set of :math:`X_j`, meaning the remaining features in :math:`\mathbf{X}` not included in :math:`X_j` for the :math:`i`-th instance.
- :math:`n` is the number of observations in the dataset.

For two features, :math:`X_j` and :math:`X_k`, the partial dependence is given by:

.. math::

   \text{PD}(X_j, X_k) = \frac{1}{n} \sum_{i=1}^{n} f(X_j, X_k, \mathbf{X}_{C_i})

This results in a 2D surface plot (or contour plot) that shows how the predicted outcome changes as the values of :math:`X_j` and :math:`X_k` vary, while the effects of the other features are averaged out.

- **Single Feature PDP:** When plotting :math:`\text{PD}(X_j)`, the result is a 2D line plot showing the marginal effect of feature :math:`X_j` on the predicted outcome, averaged over all possible values of the other features.
- **Two Features PDP:** When plotting :math:`\text{PD}(X_j, X_k)`, the result is a 3D surface plot (or a contour plot) that shows the combined marginal effect of :math:`X_j` and :math:`X_k` on the predicted outcome. The surface represents the expected value of the prediction as :math:`X_j` and :math:`X_k` vary, while all other features are averaged out.


**3D Partial Dependence Plots**

For a more comprehensive analysis, especially when exploring interactions between two features, :ref:`3D Partial Dependence Plots <3D_Partial_Dependence_Plots>` are invaluable. The partial dependence function for two features in a 3D context is:

.. math::

   \text{PD}(X_j, X_k) = \frac{1}{n} \sum_{i=1}^{n} f(X_j, X_k, \mathbf{X}_{C_i})

Here, the function :math:`f(X_j, X_k, \mathbf{X}_{C_i})` is evaluated across a grid of values for :math:`X_j` and :math:`X_k`. The resulting 3D surface plot represents how the model's prediction changes over the joint range of these two features.

The 3D plot offers a more intuitive visualization of feature interactions compared to 2D contour plots, allowing for a better understanding of the combined effects of features on the model's predictions. The surface plot is particularly useful when you need to capture complex relationships that might not be apparent in 2D.

- **Feature Interaction Visualization:** The 3D PDP provides a comprehensive view of the interaction between two features. The resulting surface plot allows for the visualization of how the model’s output changes when the values of two features are varied simultaneously, making it easier to understand complex interactions.
- **Enhanced Interpretation:** 3D PDPs offer enhanced interpretability in scenarios where feature interactions are not linear or where the effect of one feature depends on the value of another. The 3D visualization makes these dependencies more apparent.
