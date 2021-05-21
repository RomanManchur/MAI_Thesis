# MAI_Thesis

Provided package is set of tools that allow perform automated analysis of complex astronomical objects.
It expects input data to be in OIFITS format and provide following features:
 - data visualization: shows 2D plots of Squared Visibilities and Closure Phases
 - data pre-processing: through respective libraries it pre-formats input data performing quantization, normalization, interpolation.
 - find closest N-neighbours between the targets in dataset
 - builds K-means clustering model to represent groups in data.
 - builds Hierarchical clustering model to represent groups in data.
 - builds DBA k-means cluster model to represent groups in data.
 
 ### File structure
  1. wrapper.sh - a wrapper shall script to exectute main code. / Examples of supported commands are in that file/ 
  2. run.py - main execution file. 
  3. Celestial  - package that provides pogo class to save celestial object information.
  4. file_preprocessing - package that responsible for data transformation
  5. eval_distance - provides libraries for DTW distance calcualation and variants.
  6. knn - K-nn custom implementation package. 
  7. data_visualization - package that provides methods to plot data.
  
  ### Data folder structure
   1. csv - default directory to save processed data 
   2. pdf - default directory to store plots
   3. points_to_check - default directory that must contain target list file, e.i: objects to run analysis on. 
