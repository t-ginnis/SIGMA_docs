Getting Started
===============

.. _requirements:

Installing ``conda``
------------
Before using SIGMA2, it is necessary to install a python package manager such as `conda <https://www.anaconda.com/docs/main>`_.




This page describes one method for installing and using ``conda``, which will allow for SIGMA to be installed.

#. Download the appropriate **miniconda** installer for your machine from `here <https://www.anaconda.com/download/success>`_.
#. Run the installer - the default options should provide a suitable install for using SIGMA.
#. Once the installer has finished, open the conda Command Line Interface (CLI)

   * On windows, this is the Anaconda Prompt application, which should be in the start menu

   * On MacOS / Linux, this is the standard Terminal application

#. The terminal should look like (depending on your machine)

   .. code-block:: bash

      (base) USERNAME@machine : 
   
#. Run the following command to verify that `conda` is installed correctly :

   .. code-block:: bash

      conda --version

   
   It should return something like:

   
   .. code-block:: bash

      conda 25.3.1
   

.. _setup_env

Setting up a ``conda`` environment
------------

Create a new ``conda`` environment by doing the following:

#. Open the conda Command Line Interface (anaconda prompt on windows, terminal on MacOS / Linux)
#. Run the following command to create a new environment to run SIGMA2:

.. code-block:: bash

   conda create -n sigma2 python=3.10

#. Run the following to activate the environment:

.. code-block:: bash

   conda activate sigma2

Your terminal should now look like:

.. code-block:: bash

   (sigma2) USERNAME@machine : 






.. _download:

Downloading SIGMA
------------

To download the latest verision of SIGMA:

#. Go to the `SIGMA2 GitHub <https://github.com/NanoPaleoMag/SIGMA2.git>`_.
#. Press the green '<>code' drop-down menu
#. Click 'Download ZIP
#. Once the .`zip` file is downloaded, extract it to a sensible location on your machine


Alternatively, if `git` is configured to run from the command line, SIGMA2 can be downloaded from the command line using:

.. code-block:: bash

   git clone https://github.com/NanoPaleoMag/SIGMA2.git


.. _Installation:

Installing SIGMA
------------

To finish the installation of SIGMA:

#. Navigate to the ``SIGMA2`` folder on your machine that was donwloaded / extracted from the `SIGMA2 GitHub <https://github.com/NanoPaleoMag/SIGMA2.git>`_.
#. Open this folder in the ``conda`` Command Line Interface
#. Ensure that the ``sigma2`` environment is active (if it is not already) by running:

.. code-block:: bash

   conda activate sigma2

#. Install the required python packages by running the following command:

.. code-block:: bash

   pip install -r requirements.txt

SIGMA2 should now be installed correctly

.. _verification:

Verifying the Install
------------

The install can be verified by running the first cells of a tutorial notebook.

#. Open the Command Line Interface in the ``SIGMA2`` folder and activate the ``sigma2`` environment with:

.. code-block:: bash

   conda activate sigma2

#. Start a jupyter lab with

.. code-block:: bash

   jupyter lab

#. Jupyter lab should open in browser. Once it does, open the tutorial notebook in the tutorials folder
#. Run the first cell (that starts with ``from umap import UMAP # for UMAP latent space projections``) by clicking on this cell and pressing ``Shift`` + ``Enter``
#. An asterisk (*) appears next to the cell while it is running
#. If the cell runs correctly, the asterisk is replaced with [1]. If this has happened - congratulations! SIGMA2 is installed correctly



