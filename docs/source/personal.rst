Performing your own Analysis
============================

Directory Structure
-------------------
SIGMA2 is still in development, and the tutorials make use of relative imports to load SIGMA. In order to perform this analysis for your own data, it is recommended to do the following:

#. Create a folder in the SIGMA2 with a sensible name
#. In this folder, create a new jupyter notebook
#. For the first cell of this notebook, copy and run the the following cell first:

.. code-block:: python

    from umap import UMAP # for UMAP latent space projections
    import sys # for relative imports of sigma
    sys.path.insert(0,"..")
    from sigma.utils import normalisation as norm 
    from sigma.utils import visualisation as visual
    from sigma.utils.load import *
    from sigma.utils.load
    from sigma.src.utils import same_seeds
    from sigma.src.dim_reduction import Experiment
    from sigma.models.autoencoder import AutoEncoder
    from sigma.src.segmentation import PixelSegmenter
    from sigma.gui import gui
    from sigma.utils.loadtem import TEMDataset


After running this, the packages should be imported, and you can proceed with data analysis as demonstrated in the tutorial notebooks.
