from __future__ import absolute_import
from __future__ import print_function

from abc import ABC, abstractmethod

class Helper(ABC):

    @abstractmethod
    def setup(self, model):
        """Initialize something to be helpful for computing the score.
        For example, if you want to consider compression ration in your scoring scheme,
        you can define some information here such as the original model size.

        """

    @abstractmethod
    def train(self, model, callbacks):
        """Train the input model.

        """

    @abstractmethod
    def evaluate(self, model):
        """Evaluate the input model.

        """

    @abstractmethod
    def sample_training_data(self, nsamples):
        """Returns random samples from training data.

        """

    @abstractmethod
    def score(self, model):
        """Compute the score of the model in compression aspect.

        """
