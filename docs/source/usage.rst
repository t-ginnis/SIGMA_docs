Usage
=====

.. _requirements:

Requirements
------------
Before using SIGMA2, it is necessary to install a python package manager such as `conda <https://www.anaconda.com/docs/main>`_.


Installing `conda`
^^^^^^^^^^^^^
This page describes one method for installing and using `conda`, which will allow for SIGMA to be installed.

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
   





.. _installation:

Installing SIGMA
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

