## Regression benchmarking package

Code for testing the accuracy and speed of various implementations of ridge regression on widefield calcium imaging data. Currently requires a 'demo_model.mat' file in the session directory, as outputted by Max Melin's [ridge-model-matlab-baseline](https://github.com/mdmelin/ridge-model-matlab-baseline) package. You can download an example one [here](https://drive.google.com/file/d/1JT7VCApGdOhRStd-yWtgCFuFnCx-YvV9/view?usp=sharing) (session 09-Aug-2018 from animal mSM63).

Run instructions:

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python.
2. Install a Python IDE that supports Jupyter notebooks. I suggest [Visual Studio Code](https://code.visualstudio.com/download). 
3. Download or clone this package.
3. Open Anaconda Prompt.
3. Change directory to the package folder with `cd regbench`
4. Create the regbench environment with `conda env create -n regbench --file environment.yml`.
5. Open pipeline.ipynb
6. Place the 'demo_model.mat' file into a directory of the structure 'local_disk\animal\SpatialDisc\rec'.
7. Make sure the parameters at the beginning of the pipeline match the directory structure.
8. Run all cells

