import unittest
import pyccalix


class TestParamTrans(unittest.TestCase):
    # Set transformations. Both curves have to vary with the position,
    # otherwise they are constant and carry no positional information for
    # the intersection to recover.
    transformations = [
        pyccalix.Collection2D.TransformationMap() for _ in range(2)]
    for j in range(5):
        # curve 0 value at position j: 1 * 5 + (-1 - j) = 4 - j
        transformations[0].set([j], pyccalix.Linear([1.0], -1.0 - j))
        # curve 1 value at position j: 1 * 6 + (j - 5) = j + 1
        transformations[1].set([j], pyccalix.Linear([1.0], j - 5.0))
    # define collection
    collection = pyccalix.Collection2D(transformations)

    def test_apply(self):
        model = [5.0, 6.0]
        # In the common frame curve 0 is y = 4 - x and curve 1, which is
        # used transposed, is y = x - 1. They cross at (x, y) = (2.5, 1.5),
        # which is returned as [y, x].
        self.assertEqual(self.collection.apply(model), [1.5, 2.5])


if __name__ == '__main__':
    unittest.main()
