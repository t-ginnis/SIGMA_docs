Tutorials
=========

.. _introduction_to_jupyter

Introduction to Jupyter
------------------------

Using SIGMA2 does not require any previous knowledge of python or jupyter notebooks. This section contains some tips and shortcuts to help users who are less experienced with python and jupyter to run SIGMA.

Opening a Notebook
^^^^^^^^^^^^^^^^^^

In your conda command line interface, you can start jupyter lab by running the following:


.. code-block:: bash

   jupyter lab

This should open a browser window with jupyter lab running. It should look like this.

.. image:: jupyter_interface.png
  :width: 400
  :alt: Browser window showing jupyter lab running

In the left hand panel, you can see a folder structure. In here you can navigate through folders on your computer, into a folder containing the notebook(s) you wish to run.

You can then open one of these notebooks by double clicking it. It should then open to the right of this panel containing the directories.


Running Cells
^^^^^^^^^^^^^

A jupyter notebook is made up of cells. An example is shown here.

.. image:: jupyter_example.png
  :width: 400
  :alt: Example of cells in jupyter

A jupyter notebook will typically contain

* Markdown cells. These usually contain text, which is intended to provide information / guidance to the user. They do not impact how the code is run in the notebook.
* Code cells. These contain code, which will be ran by the notebook
* Outputs. These appe

To run a cell, select it with your mouse and press ``Shift`` + ``Enter``. This will run the cell and advance to the next cell.


Useful Shortcuts
^^^^^^^^^^^^^^^^

Some useful shortcuts for running and using cells in jupyter are:

* ``Shift`` + ``Enter`` will run the current cell and advance to the next cell
* ``Ctrl`` or ``Cmd`` + ``Enter`` will run the current cell without advancing to the next cell
* ``Alt`` + ``Enter`` will run the current cell, and create a new cell beneath the current cell
* Typing part of the name of an object or function, then pressing ``Tab`` will autocomplete it
* After typing a function, pressing ``Shift`` + ``Tab`` will bring up the docstring for the function. This will contain brief information about what the function does, and what arguments it expects.





.. _interactive_sem_tutorial

Interactive SEM Tutorial
------------------------

Introduction
^^^^^^^^^^^^

This tutorial is designed to walk-through the key features of SIGMA2, and demonstrates the workflow that can be used to analyse EDS data with the tools provided.

In general, the philosopy behind the SIGMA workflow is:

#. Reduce the dimensionality of the dataset into a **latent space**.
#. Produce **clusters** by grouping points in the latent space together.
#. Perform Non-negative Matrix Factorisation **(NMF)** on these clusters to determine the constituent phases that make up the sample

Opening the Notebook
^^^^^^^^^^^^^^^^^^^^

Start jupyter lab by running the following in the ``sigma2`` environment in the ``conda`` terminal in the ``SIGMA2`` folder.

.. code-block:: bash

   jupyter lab


A browser window should open, with jupyter lab running. Navigate to the ``Interactive_SEM_tutorial.ipynb`` notebook in the tutorials folder, and open it. 

Importing the Required Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first code cells of the notebook should look like:

.. code-block:: python 

   from umap import UMAP # for UMAP latent space projections
   import sys # for relative imports of sigma


.. code-block:: python 
   sys.path.insert(0,"..")
   from sigma.utils import normalisation as norm 
   from sigma.utils import visualisation as visual
   from sigma.utils.load import SEMDataset
   from sigma.src.utils import same_seeds
   from sigma.src.dim_reduction import Experiment
   from sigma.models.autoencoder import AutoEncoder
   from sigma.src.segmentation import PixelSegmenter
   from sigma.gui import gui



Running these cells will:

* Import the ``umap`` package which is needed to perform latent space projections
* Import the `sys` package -this is needed to perform "relative imports" of SIGMA2


.. note::
   SIGMA2 is still in development, so for the time being it is imported using relative imports. When you create your own notebooks for analysis, you will need to ensure the relative imports "point" to the correct place - see :doc:`personal` section for more information about using SIGMA for your own data analysis.


Loading a Dataset
^^^^^^^^^^^^^^^^^

To load a file into SIGMA, it should be loaded into the appropriate object.

* For SEM+EDS datasets, where a spectrum is available for each pixel, use the ``SEMDataset`` object
* For STEM+EDS datasets, use the ``TEMDataset`` object
* For datasets where only images of elemental maps exist, use the ``ImageDataset`` object - note that this object has more limited functionality. 


SIGMA2 supports loading from the following formats:

* Bruker Composite Format ``.bcf``
* ``.emi`` and ``.ser`` file pairs, created by FEI / ThermoFisher software
* ``.hspy`` files from Hyperspy
* ``.raw`` and ``.rpl`` files from Oxford Instruments 
* ``.h5oina`` files from Oxford Instruments

.. note::
   Certain formats, in particular the ``.raw`` and ``.rpl`` formats, may contain issues with calibration. See the tutorial notebook for these specific files to determine how to calibrate these files in SIGMA.

The example included in the dataset is a ``.bcf`` file from an SEM, so is loaded with the following cell:

.. code-block:: python 

   file_path = 'test.bcf'
   sem = SEMDataset(file_path)


We now have an ``SEMDataset`` object called ``sem`` which contains the EDS file. 

If you collected a dataset on a TEM, you will instead need to load it using the ``TEMDataset`` object, with:

.. code-block:: python

   file_path = 'tem_dataset.bcf' #replace this with the path to your tem file
   tem = TEMDataset(file_path)

Viewing the Dataset
^^^^^^^^^^^^^^^^^^^

The dataset can be viewed by running the following cell:

.. code-block:: python 

   gui.view_dataset(sem)


This cell may take some time to run, depending on the size of the dataset. Once this cell has finished running, a number will replace the ``[*]`` that appears next to the cell, and a Graphical User Interface (GUI) will appear beneath the cell. It should contain the following:

#. A text box, with the label "Energy (keV)". This can be used to search for X-Ray peaks of a given energy, if certain peaks are not already included in the metadata
#. A text box labelled "Feature List". For the ``test.bcf`` file, this box should already contain X-Ray lines, starting with "Al_Ka", though more can be added if desired.
#. A windowed GUI containing tabs labelled "Navigation Signal", "Sum Spectrum" and "Elemental Maps (raw)"

The firt tab in the gui show the navigation image. For the file used in this notebook, this is the Back Scattered Electron (BSE) image that is part of the ``test.bcf`` file.

The second tab shows the summed spectrum, the full EDS spectrum from all of the pixels in the dataset. The peaks defined in the feature list should me annotated on this plot

The third tab shows "raw" elemental map, found by integrating the peaks defined in the feature list from the centre of the peak to plus/minus the full width half maximum of the peak. The maps are "raw", as they are from the unbinned dataset.


Pre-Processing Steps
^^^^^^^^^^^^^^^^^^^^

The following cell performs some pre-processing steps to improve the projection into latent space. This involves:

#. Binning the data, reducing the total number of pixels and increasing the X-ray counts for each binned pixel
#. Normalising the data for each pixel, so that the integrated intensity is 1
#. Removing the intense zero energy peak

The degree of binning is defined by the ``rebin_signal((nx,ny))`` function, which will bin the dataset by factors of ``nx`` and ``ny`` in the x and y dimensions respectively.

The normalisation is performed with the ``peak_intensity_normalisation`` function.

The zero energy peak is removed by cropping the signal, the lower limit of this crop is defined in the ``remove_first_peak(lower_limit)`` function, where ``lower_limit`` defines the minimum energy, in keV, of the cropped spectrum.

These three steps are all performed in the single cell:


.. code-block:: python 

   # Rebin both edx and bse dataset
   sem.rebin_signal(size=(3,3))
   
   # normalisation to make the spectrum of each pixel summing to 1.
   sem.peak_intensity_normalisation()
   
   # Remove the first peak until the energy of 0.1 keV
   
   sem.remove_first_peak(end=0.1)
   

The binned dataset can be visualised by running the following cell again, and navigating to the "Elemental maps (binned)" tab.

.. code-block:: python 

   gui.view_dataset(sem)



In addition to normalising the EDS spectrum for each pixel, the "feature vectors" must also be normalised. The "feature vector" for a pixel is a vector, with a length defined by the number of X-Ray lines in "feature list". The intensity of each line in the feautre vector for a pixel is the integrated intensity of that peak.

These can be normalised using one or more of the following
* neighbour averaging
* Neighbour averaging, followed by the "z score" method (which scales based on the mean and standard deviation of the intensity - see https://en.wikipedia.org/wiki/Standard_score)
* Neighbour averaging, followed by "z score", followed by the "softmax" method (see https://en.wikipedia.org/wiki/Softmax_function) - this method is useful when there are small deviations in composition away from the mean.

The effect of each of these normalisation steps can be visualised by running this cell:

.. code-block:: python 

   gui.view_pixel_distributions(sem,norm_list=[norm.neighbour_averaging,norm.zscore,norm.softmax],cmap='Reds')

For this dataset, we will use the neighbour averaging and z score methods. We perform these on the dataset by running the following cell. If you choose to use a different set of normalisation parameters, uncomment the relevant line of code in the cell.

.. code-block:: python

   #sem.normalisation([norm.neighbour_averaging])
   sem.normalisation([norm.neighbour_averaging,norm.zscore])
   #sem.normalisation([norm.neighbour_averaging,norm.zscore,norm.softmax])


Now normalisation is complete, the intensity maps can be inspected again by running this cell:

.. code-block:: python
   print('After normalisation:')
   gui.view_intensity_maps(spectra=sem.normalised_elemental_data, element_list=sem.feature_list)


Optional - Adding the BSE intensity as a feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For certain datasets, it may be desireable to include the BSE intensity as a feature, as features may be visible on this image that are not observed in the elemental maps due to a reduced spatial resolution. The BSE image can be added to the dataset by running the following cell:

.. code-block:: python
   sem.get_feature_maps_with_nav_img(normalisation=[norm.neighbour_averaging,norm.zscore]) #include any normalisation steps that were performed earlier.

When you run this cell you *must* specify the normalisation steps used for the normalisation of the feature vectors using the ``normalisation`` argument. For example, if only neigbour averaging was performed, you should instead run:

.. code-block:: python
   sem.get_feature_maps_with_nav_img(normalisation=[norm.neighbour_averaging])










