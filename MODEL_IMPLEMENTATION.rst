====================
Model Implementation
====================

**Important Notes:**

* Check out docluster/utils. There are many helper methods and data structures that might be useful. Try to use them as much as possible.
* Please use autopep8 with vertical line wrapping. This will ensure docluster has a standard across each module.

Steps to follow:
~~~~~~~~~~~~~~~~~

1) Create a file with filename being the name of your model. The filename should be all lowercase and shouldn't have any underscores.

2) Copy and paste the following template inside the python file and code away::

    from docluster.core import Model

    class YourModelName(Model):

        def __init__(self, model_name='YourModelName'):
            """
            <One-liner description of your model>

            Credits:
            --------
            This was adapted from a post/code from <Name of author (which can be you)> that can be
            found here:
            <Website of author/code/white paper>

            Authors:
            --------
            - <Your name>
            - <Another name>

            Paramaters:
            -----------
            <Any parameters>

            model_name : str
                Name of the model that will be used for printing and saving purposes.

            Attributes:
            -----------
            <Any Attributes>
            """

            # Initialize any instance variables
            self.model_name = model_name

        def fit(self, data):
            """
            <One-liner description of what this method achieves>

            Paramaters:
            -----------
            data : <type>
                The data that is going to be <what will it be used for>.

            Return:
            --------
            <Any Return values>
            """

            # Apply your model and return the most essential result of your model.
            # This is usually a vector of floats, integers or strings.
            pass


      # Create any other methods you think are useful.
      # You can have helper methods as well, but put a '_' in front of the method name
      # to make it private to the class.

3) Create a test file in tests/models/<the type of the model> with naming convention of **test_yourmodelname.py>**.

4) Copy and paste following template inside the python file and test away::

    
