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

        Usage:
        ------
        <Something like below.>

        >>> model = YourModelName()
        >>> model.fit(data)


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
