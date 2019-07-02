import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy
import warnings

class ABBA(object):
    """
    ABBA: Aggregate Brownian bridge-based approximation of time series, see [1].

    Parameters
    ----------
    tol - float/ list
        Tolerance used during compression and digitization. Accepts either float
        or a list of length two. If float given then same tolerance used for both
        compression and digitization. If list given then first element used for
        compression and second element for digitization.

    scl - float
        Scaling parameter in range 0 to infty. Scales the lengths of the compressed
        representation before performing clustering.

    min_k - int
        Minimum value of k, the number of clusters. If min_k is greater than the
        number of pieces being clustered then each piece will belong to its own
        cluster. Warning given.

    max_k - int
        Maximum value of k, the number of clusters.

    max_len - int
        Maximum length of any segment, prevents issue with growing tolerance for
        flat time series.

    verbose - 0, 1 or 2
        Whether to print details.
        0 - Print nothing
        1 - Print key information
        2 - Print all important information

    seed - True/False
        Determine random number generator for centroid initialization during
        sklearn KMeans algorithm. If True, then randomness is deterministic and
        ABBA produces same representation (with fixed parameters) run by run.

    Raises
    ------
    ValueError: Invalid tol, Invalid scl, Invalid min_k, len(pieces)<min_k.

    Example
    -------
    >>> from ABBA import ABBA
    >>> ts = [-1, 0.1, 1.3, 2, 1.9, 2.4, 1.8, 0.8, -0.5]
    >>> abba = ABBA(tol=0.5, scl=0, min_k=1, max_k = 3)
    >>> string, centers = abba.transform(ts)
    Warning: Time series does not have zero mean.
    Warning: Time series does not have unit variance.
    Compression: Reduced time series of length 9 to 3 segments
    Digitization: Using 2 symbols
    >>> reconstructed_ts = abba.inverse_transform(string, centers, ts[0])

    References
    ------
    [1] S. Elsworth and S. Güttel. ABBA: Aggregate Brownian bridge-based
    approximation of time series, MIMS Eprint 2019.11
    (http://eprints.maths.manchester.ac.uk/2712/), Manchester
    Institute for Mathematical Sciences, The University of Manchester, UK, 2019.
    """

    def __init__(self, tol=0.1, scl=0, min_k=2, max_k=100, max_len = np.inf, verbose=1, seed=True):
        self.tol = tol
        self.scl = scl
        self.min_k = min_k
        self.max_k = max_k
        self.max_len = max_len
        self.verbose = verbose
        self.seed = seed
        self._check_parameters()

        # Import Cpp wrapper
        Ck = False
        if self.scl == 0 or self.scl == np.inf:
            try:
                from src.Ckmeans import kmeans_1d_dp
                from src.Ckmeans import double_vector
                Ck = True
            except:
                warnings.warn('Ckmeans module unavailable, try running makefile. Using sklearn KMeans instead.',  stacklevel=3)
        self.Ck = Ck

    def transform(self, time_series):
        """
        Convert time series representation to ABBA symbolic representation

        Parameters
        ----------
        time_series - numpy array
            Normalised time series as numpy array.

        Returns
        -------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.

        centers - numpy array
            Centres of clusters from clustering algorithm. Each center corresponds
            to character in string.

        """
        time_series_ = self._check_time_series(time_series)
        # Perform compression
        pieces = self.compress(time_series_)

        # Perform digitization
        string, centers = self.digitize(pieces)
        return string, centers

    def _check_time_series(self, time_series):
        # Convert time series to numpy array
        time_series_ = np.array(time_series)

        # Check normalisation if Normalise=False and Verbose
        if self.verbose == 2:
            if np.mean(time_series_) > np.finfo(float).eps:
                print('Warning: Time series does not have zero mean.')
            if np.abs(np.std(time_series_) - 1) > np.finfo(float).eps:
                print('Warning: Time series does not have unit variance.')
        return time_series_

    def _check_parameters(self):
        self.compression_tol = None
        self.digitization_tol = None

        # Check tol
        if isinstance(self.tol, list) and len(self.tol) == 2:
            self.compression_tol, self.digitization_tol = self.tol
        elif isinstance(self.tol, list) and len(self.tol) == 1:
            self.compression_tol = self.tol[0]
            self.digitization_tol = self.tol[0]
        elif isinstance(self.tol, float):
            self.compression_tol = self.tol
            self.digitization_tol = self.tol
        else:
            raise ValueError('Invalid tol.')

        # Check scl (scaling parameter)
        if self.scl < 0:
            raise ValueError('Invalid scl.')

        # Check min_k and max_k
        if self.min_k > self.max_k:
            raise ValueError('Invalid limits: min_k must be less than or equal to max_k')

        if self.verbose not in [0, 1, 2]:
            self.verbose == 1 # set to default
            print('Invalid verbose, setting to default')



    def inverse_transform(self, string, centers, start=0):
        """
        Convert ABBA symbolic representation back to numeric time series representation.

        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.

        centers - numpy array
            Centers of clusters from clustering algorithm. Each center corresponds
            to character in string.

        start - float
            First element of original time series. Applies vertical shift in
            reconstruction. If not specified, the default is 0.

        Returns
        -------
        times_series - list
            Reconstruction of the time series.
        """

        pieces = self.inverse_digitize(string, centers)
        pieces = self.quantize(pieces)
        time_series = self.inverse_compress(start, pieces)
        return time_series

    def compress(self, time_series):
        """
        Approximate a time series using a continuous piecewise linear function.

        Parameters
        ----------
        time_series - numpy array
            Time series as numpy array.

        Returns
        -------
        pieces - numpy array
            Numpy array with three columns, each row contains length, increment
            error for the segment.
        """
        start = 0 # start point
        end = 1 # end point
        pieces = np.empty([0, 3]) # [increment, length, error]
        tol = self.compression_tol**2
        x = np.arange(0, len(time_series))
        epsilon =  np.finfo(float).eps

        while end < len(time_series):
            # error function for linear piece
            inc = time_series[end] - time_series[start]
            err = np.linalg.norm((time_series[start] + (inc/(end-start))*x[0:end-start+1]) - time_series[start:end+1])**2

            if (err <= tol*(end-start-1) + epsilon) and (end-start-1 < self.max_len):
            # epsilon added to prevent error when err ~ 0 and (end-start-1) = 0
                (lastinc, lasterr) = (inc, err)
                end += 1
                continue
            else:
                pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
                start = end - 1

        pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
        if self.verbose in [1, 2]:
            print('Compression: Reduced time series of length', len(time_series), 'to', len(pieces), 'segments')
        return pieces

    def inverse_compress(self, start, pieces):
        """
        Reconstruct time series from its first value `ts0` and its `pieces`.
        `pieces` must have (at least) two columns, incremenent and window width, resp.
        A window width w means that the piece ranges from s to s+w.
        In particular, a window width of 1 is allowed.

        Parameters
        ----------
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction.

        pieces - numpy array
            Numpy array with three columns, each row contains increment, length,
            error for the segment. Only the first two columns are required.

        Returns
        -------
        time_series : Reconstructed time series
        """
        time_series = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            y = time_series[-1] + x
            time_series = time_series + y[1:].tolist()
        return time_series

    def _max_cluster_var(self, pieces, labels, centers, k):
        """
        Calculate the maximum variance among all clusters after k-means, in both
        the inc and len dimension.

        Parameters
        ----------
        pieces - numpy array
            One or both columns from compression. See compression.

        labels - list
            List of ints corresponding to cluster labels from k-means.

        centers - numpy array
            centers of clusters from clustering algorithm. Each center corresponds
            to character in string.

        k - int
            Number of clusters. Corresponds to numberof rows in centers, and number
            of unique symbols in labels.

        Returns
        -------
        variance - float
            Largest variance among clusters from k-means.
        """
        d1 = [0] # direction 1
        d2 = [0] # direction 2
        for i in range(k):
            matrix = ((pieces[np.where(labels==i), :] - centers[i])[0]).T
            # Check not all zero
            if not np.all(np.abs(matrix[0,:]) < np.finfo(float).eps):
                # Check more than one value
                if len(matrix[0,:]) > 1:
                    d1.append(np.var(matrix[0,:]))

            # If performing 2-d clustering
            if matrix.shape[0] == 2:
                # Check not all zero
                if not np.all(np.abs(matrix[1,:]) < np.finfo(float).eps):
                    # Check more than one value
                    if len(matrix[1,:]) > 1:
                        d2.append(np.var(matrix[1,:]))
        return np.max(d1), np.max(d2)

    def _build_centers(self, pieces, labels, c1, k, col):
        """
        utility function for digitize, helps build 2d cluster centers after 1d clustering.

        Parameters
        ----------
        pieces - numpy array
            Time series in compressed format. See compression.

        labels - list
            List of ints corresponding to cluster labels from k-means.

        c1 - numpy array
            1d cluster centers

        k - int
            Number of clusters

        col - 0 or 1
            Which column was clustered during 1d clustering

        Returns
        -------
        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
        """
        c2 = []
        for i in range(k):
            location = np.where(labels==i)[0]
            if location.size == 0:
                c2.append(np.NaN)
            else:
                c2.append(np.mean(pieces[location, col]))
        if col == 0:
            return (np.array((c2, c1))).T
        else:
            return (np.array((c1, c2))).T

    def digitize(self, pieces):
        """
        Convert compressed representation to symbolic representation using clustering.

        Parameters
        ----------
        pieces - numpy array
            Time series in compressed format. See compression.

        Returns
        -------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.

        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
        """
        # Check number of pieces
        if len(pieces) < self.min_k:
            raise ValueError('Number of pieces less than min_k.')

        # Import c++ functions
        if self.Ck:
            from src.Ckmeans import kmeans_1d_dp
            from src.Ckmeans import double_vector

        # Initialise variables
        centers = np.array([])
        labels = []

        # construct tol_s
        s = .20
        N = 1
        for i in pieces:
            N += i[0]
        bound = ((6*(N-len(pieces)))/(N*len(pieces)))*((self.digitization_tol*self.digitization_tol)/(s*s))

        data = deepcopy(pieces[:,0:2])

        # scale length to unit variance
        if self.scl != 0:
            len_std = np.std(pieces[:,0])
            len_std = len_std if len_std > np.finfo(float).eps else 1
            data[:,0] /= len_std

        # scale inc to unit variance
        if self.scl != np.inf:
            inc_std = np.std(pieces[:,1])
            inc_std = inc_std if inc_std > np.finfo(float).eps else 1
            data[:,1] /= inc_std

        # Select first column and check unique for Ckmeans
        if self.scl == np.inf:
            data = data[:,0]
            if self.Ck and (len(set(data)) < self.min_k):
                warnings.warn('Note enough unique pieces for Ckmeans. Using sklearn KMeans instead.',  stacklevel=3)
                self.Ck = False

        # Select second column and check unique for Ckmeans
        if self.scl == 0:
            data = data[:,1]
            if self.Ck and (len(set(data)) < self.min_k):
                warnings.warn('Note enough unique pieces for Ckmeans. Using sklearn KMeans instead.',  stacklevel=3)
                self.Ck = False

        # Use Ckmeans
        if self.Ck:
            d = double_vector(data)
            output = kmeans_1d_dp(d, self.min_k, self.max_k, bound, 'linear')
            labels = np.array(output.cluster)

            c = np.array(output.centres)
            if self.scl == np.inf:
                c *= len_std
                centers = self._build_centers(pieces, labels, c, output.Kopt, 1)
            else:
                c *= inc_std
                centers = self._build_centers(pieces, labels, c, output.Kopt, 0)

            if self.verbose in [1, 2]:
                print('Digitization: Using', output.Kopt, 'symbols')

        # Use Kmeans
        else:
            if self.scl == np.inf:
                data = data.reshape(-1,1) # reshape for sklearn
            elif self.scl == 0:
                data = data.reshape(-1,1) # reshape for sklearn
            else:
                data[:,0] *= self.scl # scale lengths accordingly

            # Run through values of k from min_k to max_k checking bound
            if self.digitization_tol != 0:
                error = np.inf
                k = self.min_k - 1
                while k <= self.max_k-1 and (error > bound):
                    k += 1
                    # tol=0 ensures labels and centres coincide
                    if self.seed:
                        kmeans = KMeans(n_clusters=k, tol=0, random_state=0).fit(data)
                    else:
                        kmeans = KMeans(n_clusters=k, tol=0).fit(data)
                    centers = kmeans.cluster_centers_
                    labels = kmeans.labels_
                    error_1, error_2 = self._max_cluster_var(data, labels, centers, k)
                    error = max([error_1, error_2])
                    if self.verbose == 2:
                        print('k:', k)
                        print('d1_error:', error_1, 'd2_error:', error_2, 'bound:', bound)
                if self.verbose in [1, 2]:
                    print('Digitization: Using', k, 'symbols')

            # Zero error so cluster with largest possible k.
            else:
                if len(data) < self.max_k:
                    k = len(data)
                else:
                    k = self.max_k

                # tol=0 ensures labels and centres coincide
                kmeans = KMeans(n_clusters=k, tol=0).fit(data)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                error = self._max_cluster_var(data, labels, centers, k)
                if self.verbose in [1, 2]:
                    print('Digitization: Using', k, 'symbols')

            # build cluster centers
            c = centers.reshape(1,-1)[0]
            if self.scl == np.inf:
                c *= len_std
                centers = self._build_centers(pieces, labels, c, k, 1)
            elif self.scl == 0:
                c *= inc_std
                centers = self._build_centers(pieces, labels, c, k, 0)
            else:
                centers[:,0] *= len_std
                centers[:,0] /= self.scl # reverse scaling
                centers[:,1] *= inc_std

        # Convert labels to string
        string = ''.join([ chr(97 + j) for j in labels ])
        return string, centers

    def inverse_digitize(self, string, centers):
        """
        Convert symbolic representation back to compressed representation for reconstruction.

        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.

        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.


        Returns
        -------
        pieces - np.array
            Time series in compressed format. See compression.
        """
        pieces = np.empty([0,2])
        for p in string:
            pc = centers[ord(p)-97,:]
            pieces = np.vstack([pieces, pc])
        return pieces

    def quantize(self, pieces):
        """
        Realign window lengths with integer grid.

        Parameters
        ----------
        pieces: Time series in compressed representation.

        Returns
        -------
        pieces: Time series in compressed representation with window length adjusted to integer grid.
        """
        if len(pieces) == 1:
            pieces[0,0] = round(pieces[0,0])
        else:
            for p in range(len(pieces)-1):
                corr = round(pieces[p,0]) - pieces[p,0]
                pieces[p,0] = round(pieces[p,0] + corr)
                pieces[p+1,0] = pieces[p+1,0] - corr
                if pieces[p,0] == 0:
                    pieces[p,0] = 1
                    pieces[p+1,0] -= 1
            pieces[-1,0] = round(pieces[-1,0])
        return pieces
