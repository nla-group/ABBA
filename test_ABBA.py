import unittest
from ABBA import ABBA
import numpy as np
import warnings

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test

class test_ABBA(unittest.TestCase):
    #--------------------------------------------------------------------------#
    # _check_parameters
    #--------------------------------------------------------------------------#
    def test_CheckParameters_TolFloat(self):
        """
        tolerance should be float not integer
        """
        self.assertRaises(ValueError, ABBA, tol=1)

    def test_CheckParameters_TolList(self):
        """
        tolerance should be list, maximum size 2
        """
        self.assertRaises(ValueError, ABBA, tol=[1.0, 1.0, 1.0])

    def test_CheckParameters_SclPositive(self):
        """
        Scaling parameter should be >=0
        """
        self.assertRaises(ValueError, ABBA, scl=-0.1)

    def test_CheckParameters_KBounds(self):
        """
        min_k and max_k bounds should be such that min_k < max_k
        """
        self.assertRaises(ValueError, ABBA, min_k=6, max_k=3)

    #--------------------------------------------------------------------------#
    # compress
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_Compress_tslength2(self):
        """
        Test compression when time series given is of length 2
        """
        ts = [1, 3]
        abba = ABBA(verbose=0)
        pieces = abba.compress(ts)
        self.assertTrue(np.allclose(np.array([[1.0,2.0,0.0]]), pieces))

    @ignore_warnings
    def test_Compress_Flatline(self):
        """
        Test compression on a flat time series
        """
        ts = [1]*100
        abba = ABBA(verbose=0)
        pieces = abba.compress(ts)
        self.assertTrue(np.allclose(np.array([[99,0.0,0.0]]), pieces))

    @ignore_warnings
    def test_Compress_NoCompression(self):
        """
        Test compression on time series where tolerance so small that no compression
        is achieved
        """
        ts = [1, -1]*50
        abba = ABBA(verbose=0)
        pieces = abba.compress(ts)
        correct_pieces = [[1, -2, 0], [1, 2, 0]]*49
        correct_pieces += [[1, -2, 0]]
        correct_pieces = np.array(correct_pieces)
        self.assertTrue(np.allclose(correct_pieces, pieces))

    @ignore_warnings
    def test_Compress_Norm2(self):
        """
        Test compression with norm = 2
        """
        ts = [0, 2, 3, 2, 4, -1, 0, -1, 1, 0, -4, 0]
        abba = ABBA(tol=2.0, verbose=0)
        pieces = abba.compress(ts)
        correct_pieces = [[4, 4, 3],
                          [1, -5, 0],
                          [4, 1, 38/16],
                          [1, -4, 0],
                          [1, 4, 0]]
        correct_pieces = np.array(correct_pieces)
        self.assertTrue(np.allclose(correct_pieces, pieces))

    @ignore_warnings
    def test_Compress_Norm1(self):
        """
        Test compression with norm = 1
        """
        ts = [0, 2, 3, 2, 4, -1, 0, -1, 1, 0, -4, 0]
        abba = ABBA(tol=2.0, verbose=0, norm=1)
        pieces = abba.compress(ts)
        correct_pieces = [[4, 4, 3],
                          [1, -5, 0],
                          [4, 1, 5/2],
                          [1, -4, 0],
                          [1, 4, 0]]
        correct_pieces = np.array(correct_pieces)
        self.assertTrue(np.allclose(correct_pieces, pieces))

    #--------------------------------------------------------------------------#
    # inverse_compress
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_InverseCompress_OnePiece(self):
        """
        Test inverse_compress with only one piece
        """
        abba = ABBA(verbose=0)
        pieces = np.array([[1,4.0,0]])
        ts = abba.inverse_compress(0, pieces)
        correct_ts = np.array([0, 4])
        self.assertTrue(np.allclose(ts, correct_ts))

    @ignore_warnings
    def test_InverseCompress_Example(self):
        """
        Test inverse_compress on generic example
        """
        pieces = [[4, 4, 3],
                  [1, -5, 0],
                  [4, 1, 5/2],
                  [1, -4, 0],
                  [1, 4, 0]]
        pieces = np.array(pieces)
        abba = ABBA(verbose=0)
        ts = abba.inverse_compress(0, pieces)
        correct_ts = np.array([0, 1, 2, 3, 4, -1, -3/4, -2/4, -1/4, 0, -4, 0])
        self.assertTrue(np.allclose(ts, correct_ts))

    #--------------------------------------------------------------------------#
    # digitize
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_Digitize_ExampleScl0(self):
        """
        Test digitize function on same generic example with scl = 0
        """
        abba = ABBA(scl=0, verbose=0, seed=True)
        pieces = [[4, 4, 3],
                  [1, -5, 0],
                  [4, 1, 5/2],
                  [1, -4, 0],
                  [1, 4, 0]]
        pieces = np.array(pieces)
        string, centers = abba.digitize(pieces)
        correct_centers = np.array([[3, 3], [1, -9/2]])
        self.assertTrue(all([string=='ababa', np.allclose(centers, correct_centers)]))

    @ignore_warnings
    def test_Digitize_ExampleScl1(self):
        """
        Test digitize function on same generic example with scl = 1
        """
        abba = ABBA(scl=1, verbose=0, seed=True)
        pieces = [[4, 4, 3],
                  [1, -5, 0],
                  [4, 1, 5/2],
                  [1, -4, 0],
                  [1, 4, 0]]
        pieces = np.array(pieces)
        string, centers = abba.digitize(pieces)
        correct_centers = np.array([[4, 5/2], [1, -9/2], [1, 4]])
        self.assertTrue(all([string=='ababc', np.allclose(centers, correct_centers)]))

    @ignore_warnings
    def test_Digitize_ExampleSclInf(self):
        """
        Test digitize function on same generic example with scl = inf
        """
        abba = ABBA(scl=np.inf, verbose=0, seed=True)
        pieces = [[4, 4, 3],
                  [1, -5, 0],
                  [4, 1, 5/2],
                  [1, -4, 0],
                  [1, 4, 0]]
        pieces = np.array(pieces)
        string, centers = abba.digitize(pieces)
        correct_centers = np.array([[1, -5/3], [4, 5/2]])
        self.assertTrue(all([string=='babaa', np.allclose(centers, correct_centers)]))

    @ignore_warnings
    def test_Digitize_SymbolOrdering(self):
        """
        Test digitize function orders letters by most occuring symbol.
        """
        abba = ABBA(verbose=0)
        pieces = [[1,1,0],
                  [50,50,0],
                  [100,100,0],
                  [2,2,0],
                  [51,51,0],
                  [3,3,0]]
        pieces = np.array(pieces).astype(float)
        string, centers = abba.digitize(pieces)
        self.assertTrue('abcaba'==string)

    @ignore_warnings
    def test_Digitize_OneCluster(self):
        """
        Test digitize function to make one large cluster
        """
        inc = np.random.randn(100,1)
        abba = ABBA(verbose=0, min_k=1, tol=10.0)
        pieces = np.hstack([np.ones((100,1)), inc, np.zeros((100,1))])
        string, centers = abba.digitize(pieces)
        self.assertTrue('a'*100 == string)

    @ignore_warnings
    def test_Digitize_NotEnoughPieces(self):
        """
        Test digitize function where min_k is greater than the number of pieces
        """
        abba = ABBA(verbose=0, min_k=10)
        pieces = [[4, 4, 3],
                  [1, -5, 0],
                  [4, 1, 5/2],
                  [1, -4, 0],
                  [1, 4, 0]]
        pieces = np.array(pieces)
        self.assertRaises(ValueError, abba.digitize, pieces)

    @ignore_warnings
    def test_Digitize_TooManyK(self):
        """
        Test digitize function where less than min_k are required for perfect
        clustering.
        """
        abba = ABBA(verbose=0, min_k=3, seed=True)
        pieces = [[1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]]
        pieces = np.array(pieces).astype(float)
        string, centers = abba.digitize(pieces)
        correct_centers = np.array([[1, 1], [1, 1], [1, 1]])
        self.assertTrue(all([string=='ccccc', np.allclose(centers, correct_centers)]))

    #--------------------------------------------------------------------------#
    # inverse_digitize
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_InverseDigitize_example(self):
        """
        Test inverse digitize on a generic example
        """
        abba = ABBA(verbose=0)
        centers = np.array([[3, 3], [1, -9/2]]).astype(float)
        string = 'ababa'
        pieces = abba.inverse_digitize(string, centers)
        correct_pieces = [[3, 3],
                          [1, -9/2],
                          [3, 3],
                          [1, -9/2],
                          [3, 3]]
        correct_pieces = np.array(correct_pieces).astype(float)
        self.assertTrue(np.allclose(pieces, correct_pieces))

    #--------------------------------------------------------------------------#
    # quantize
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_Quantize_NoRoundingNeeded(self):
        """
        Test quantize function on an array where no rounding is needed
        """
        pieces = [[2, 1],
                  [3, 1],
                  [4, 2],
                  [1, 2],
                  [1, -5],
                  [2, -1]]
        pieces = np.array(pieces)
        abba = ABBA(verbose=0)
        self.assertTrue(np.allclose(pieces, abba.quantize(pieces)))

    @ignore_warnings
    def test_Quantize_AccumulateError(self):
        """
        Test quantize function with distributed rounding
        """
        pieces = [[7/4, 1],
                  [7/4, 1],
                  [7/4, 1],
                  [7/4, 1],
                  [5/4, 1],
                  [5/4, 1],
                  [5/4, 1],
                  [5/4, 1]]
        pieces = np.array(pieces).astype(float)
        abba = ABBA(verbose=0)
        pieces = abba.quantize(pieces)
        correct_pieces = [[2, 1],
                          [2, 1],
                          [1, 1],
                          [2, 1],
                          [1, 1],
                          [2, 1],
                          [1, 1],
                          [1, 1]]
        self.assertTrue(np.allclose(correct_pieces, abba.quantize(pieces)))

    @ignore_warnings
    def test_Quantise_Half(self):
        """
        Test quantize function where all values are 1.5
        """
        pieces = [[3/2, 1],
                  [3/2, 1],
                  [3/2, 1],
                  [3/2, 1],
                  [3/2, 1],
                  [3/2, 1],
                  [3/2, 1],
                  [3/2, 1]]
        pieces = np.array(pieces).astype(float)
        abba = ABBA(verbose=0)
        pieces = abba.quantize(pieces)
        correct_pieces = [[2, 1],
                          [1, 1],
                          [2, 1],
                          [1, 1],
                          [2, 1],
                          [1, 1],
                          [2, 1],
                          [1, 1]]
        self.assertTrue(np.allclose(correct_pieces, abba.quantize(pieces)))

    #--------------------------------------------------------------------------#
    # _build_centers
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_BuildCenters_c1(self):
        """
        Test utility function _build_centers on column 2
        """
        pieces = [[4, 4],
                  [1, -5],
                  [4, 1],
                  [1, -4],
                  [1, 4]]
        pieces = np.array(pieces).astype(float)
        labels = np.array([0, 1, 1, 1, 0])
        k = 2
        c1 = [4,-4]
        col = 0
        abba = ABBA(verbose=0)
        c = abba._build_centers(pieces, labels, c1, k, col)
        correct_c = np.array([[5/2, 4], [2, -4]])
        self.assertTrue(np.allclose(correct_c, c))

    @ignore_warnings
    def test_BuildCenters_c2(self):
        """
        Test utility function _build_centers on column 1
        """
        pieces = [[4, 4],
                  [1, -5],
                  [4, 1],
                  [1, -4],
                  [1, 4]]
        pieces = np.array(pieces).astype(float)
        labels = np.array([0, 1, 0, 1, 1])
        k = 2
        c1 = [4,1]
        col = 1
        abba = ABBA(verbose=0)
        c = abba._build_centers(pieces, labels, c1, k, col)
        correct_c = np.array([[4, 5/2], [1, -5/3]])
        self.assertTrue(np.allclose(correct_c, c))

    #--------------------------------------------------------------------------#
    # _max_cluster_var
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_MaxClusterVar_example(self):
        """
        Test utility function _max_cluster_var
        """
        pieces = [[4, 4],
                  [1, -5],
                  [4, 1],
                  [1, -4],
                  [1, 4]]
        pieces = np.array(pieces).astype(float)
        labels = np.array([0, 0, 0, 1, 1])
        centers = np.array([[3, 0], [1, 0]]).astype(float)
        k = 2
        abba = ABBA()
        (e1, e2) = abba._max_cluster_var(pieces, labels, centers, k)
        ee1 = max([np.var([1,-2,1]), np.var([0,0])])
        ee2 = max([np.var([4,-5,1]), np.var([4,-4])])
        self.assertTrue(np.allclose([e1, e2], [ee1, ee2]))

    #--------------------------------------------------------------------------#
    # digitize_inc
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_DigitizeInc_NotWeightedNotSymmetricOneNorm(self):
        """
        Test digitize_inc with weighted=False and symmetric=False and 1 norm
        """
        pieces = [[1, -5],
                  [2, 0],
                  [1, -6],
                  [2, 2],
                  [1, -4],
                  [1, 3],
                  [4, 8]]
        pieces = np.array(pieces).astype(float)
        abba = ABBA(verbose=0, norm=1)
        string, centers = abba.digitize_inc(pieces, tol=2/3+1e-10, weighted=False, symmetric=False)
        correct_centers = [[1, -5],
                           [5/3, 2],
                           [4, 8]]
        correct_centers = np.array(correct_centers)
        self.assertTrue(np.allclose(centers, correct_centers))

    @ignore_warnings
    def test_DigitizeInc_NotWeightedNotSymmetricTwoNorm(self):
        """
        Test digitize_inc with weighted=False and symmetric=False and 2 norm
        """
        pieces = [[1, -5],
                  [2, 0],
                  [1, -6],
                  [2, 2],
                  [1, -4],
                  [1, 3],
                  [4, 8]]
        pieces = np.array(pieces).astype(float)
        abba = ABBA(verbose=0, norm=2)
        string, centers = abba.digitize_inc(pieces, tol=42/27+1e-10, weighted=False, symmetric=False)
        correct_centers = [[1, -5],
                           [5/3, 5/3],
                           [4, 8]]
        correct_centers = np.array(correct_centers)
        self.assertTrue(np.allclose(centers, correct_centers))

    @ignore_warnings
    def test_DigitizeInc_WeightedNotSymmetricOneNorm(self):
        """
        Test digitize_inc with weighted=True and symmetric=False and 1 norm
        """
        pieces = [[1, -5],
                  [2, 0],
                  [1, -6],
                  [2, 2],
                  [1, -4],
                  [1, 3],
                  [4, 8]]
        pieces = np.array(pieces).astype(float)
        abba = ABBA(verbose=0, norm=1)
        string, centers = abba.digitize_inc(pieces, tol=2/3+1e-10, weighted=True, symmetric=False)
        print(centers)
        correct_centers = [[1, -5],
                           [5/3, 5/3],
                           [4, 8]]
        correct_centers = np.array(correct_centers)
        #self.assertTrue(np.allclose(centers, correct_centers))


if __name__ == "__main__":
    unittest.main()
