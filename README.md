## Regression benchmarking package

Code for testing the accuracy and speed of various implementations of ridge regression on widefield calcium imaging data. Scalable to handle an arbitrary number of methods for calculating alpha (the ridge parameter or parameters) and for performing cross-validated prediction. Currently requires a 'demo_model.mat' file in the session directory, as outputted by Max Melin's [ridge-model-matlab-baseline](https://github.com/mdmelin/ridge-model-matlab-baseline) package. You can download an example one [here](https://drive.google.com/file/d/1JT7VCApGdOhRStd-yWtgCFuFnCx-YvV9/view?usp=sharing) (session 09-Aug-2018, animal mSM63, data from [Mussal et al. 2019](https://www.nature.com/articles/s41593-019-0502-4)).

First run only:
1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python.
2. Install a Python IDE that supports Jupyter notebooks. I suggest [Visual Studio Code](https://code.visualstudio.com/download). 
3. Download or clone this package.
4. Open Anaconda Prompt.
5. Change directory to the package folder with `cd regbench`
6. Create the regbench environment with `conda env create -n regbench --file environment.yml`
7. Open pipeline.ipynb
8. Activate the regbench environment. In Visual Studio Code, this is done by [pressing select kernel](https://code.visualstudio.com/assets/docs/datascience/jupyter/native-kernel-picker.png) in the top-right corner and then picking 'regbench' from the dropdown menu.
9. Download the recording .mat.
10. Make sure the parameters at the beginning of the pipeline match the directory and name of the recording .mat.
11. Run all cells

Subsequent runs:
1. Open pipeline.ipynb
2. Activate the regbench environment.
3. Run all cells.
