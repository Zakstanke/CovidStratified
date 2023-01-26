Files and Functions used in Covid Parameter Estimation.

How to use:
The simplest way to use this code is to use the Markdown files, as they show how each function is used one step at a time. Open CovidMarkdownRegularized.ipynb in Jupyter Notebook, make sure that it can access all of the input files/directories correctly, and then "run all cells". If you want to look at the data you generated, open StandCovid_figures.RMD in RStudio and "run all cells". All of the data tranformations will happen automatically and several figures will be created. Obtaining all the figures shown throughout the paper and supplementary information will require manipulation of some of ggplot functions.

--------------------------------------------------Data-------------------------------------------------
    cases.csv - Subsetted data from fullcases.csv, only including case counts for the 6 countries grouped by decades
    prem___.csv - Contact Matricies, made by Prem et al., for 6 different countries (CAN, CHN, GBR, ISR, ITA, NLD)
    fullcases.csv - Case count with additional information like start/end dates, test counts, etc. Not used in computation
-------------------------------------------------------------------------------------------------------


-----------------------------------------------Executables---------------------------------------------
    CovidDataCleaning - R Markdown file used for subsetting and cleaning the orignal data. No longer important for computation.
    CovidEig.py - Python file containing functions for data input and array/matrix manipulations required in analysis
    CovidMarkdownRegularized.ipynb - Markdown file for estimating parameters using regularization method. Cells show step by step instructions on how to input the data, run a basin hopping estimation, and export results to the Output folder.
    CovidReg.py - Python file containing functions specific to the regularization method. 
    StandCovid_figures.RMD - R Markdown file for creating box and whisker plots of the parameter estimation trials and heatmaps of various matricies. To get ALL of the results seen in the Output folder, you will need to modify the ggplot function calls in various ways, since this was written to explore our results visually instead of producing all the final graphs in one run.


-----Executable files below this line are included in the GitHub package for historical reference,-----
-----but not used in the final version of the paper or computation. They likely don't work anymore-----
    CI testing.ipynb - Jupyter notebook markdown file for testing the functions to extract confidence intervals for our KL parameter estimates
    CovidCI.py - Python file containing functions for extracting confidence intervals .
    CovidEM2.py - Copy of CovidKL that was used to experiment with functions in the debugging process.
    CovidKL.py - Python file containing functions for using Kullback-Leibler Divergence in the basin hopping algorithm
    CovidMarkdown.ipynb - Markdown file for estimating parameters using KL divergence. Shows step by step process on how to use functions from CovidKL
-------------------------------------------------------------------------------------------------------


--------------------------------------------------Output-----------------------------------------------
    Graphics for presentation.pptx - Powerpoint slides containing multiple variations of box and whisker plot made from parameter estimation trials. Variations were created to be used in a presentation to introduce the elements of the box plot one at a time.
    Heatmaps.pptx - Powerpoint slides containing heatmaps of SCV matrix
    Heatmaps scaled.pptx - Powerpoint slides containing heatmaps of SCV matrix after scaling with constant of proportionality. Example calculation given on first slide
    Heatmaps Transmission.pptx - Powerpoint slides containing heatmaps of transmission matrices. Constants of Proportionality and equations are given on the first slide
    Median SCVs.xlsx - Excel sheet of SCV matrices being calculated using the median values for values of S and V
    SortedFullData.xlsx - Excel sheet of optimization trials, sorted by optimization return value. Used for exploratory purposes
    temp.csv - Excel sheet containing something that I needed once. Honestly I don't remember whats in here and it probably isn't important anyway.
    ThetaIter exploration.XLSX - Excel sheet for exploring optimization results.
    ThetaIterations.csv - Excel sheet containing 200 optimization trials for parameter values of S and V. Contains the randomized starting points, the optimized values, and the function return value.
    ThetaIterationsMax.csv - Same as ThetaIterations.csv, except the regularization term has larger degrees of freedom
    ThetaIterationsMin.csv - Same as ThetaIterations.csv, except the regularization term has smaller degrees of freedom
-------------------------------------------------------------------------------------------------------