import numpy as np
import torch
import itertools
from numpy.fft import rfft2
import itertools

def dict_combiner(mydict):
    """
    Combines the values of a dictionary into a list of dictionaries,
    where each dictionary represents a combination of the values.

    Args:
        mydict (dict): The input dictionary containing keys and lists of values.

    Returns:
        list: A list of dictionaries, where each dictionary represents a combination
              of the values from the input dictionary.

    Example:
        >>> mydict = {'A': [1, 2], 'B': [3, 4]}
        >>> dict_combiner(mydict)
        [{'A': 1, 'B': 3}, {'A': 1, 'B': 4}, {'A': 2, 'B': 3}, {'A': 2, 'B': 4}]
    """
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v))
                           for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list

class UnitGaussianNormalizer(object):
    """
    A class for normalizing data using unit Gaussian normalization.

    Attributes:
        mean (numpy.ndarray): The mean values of the input data.
        std (numpy.ndarray): The standard deviation values of the input data.
        eps (float): A small value added to the denominator to avoid division by zero.

    Methods:
        encode(x): Normalize the input data using unit Gaussian normalization.
        decode(x): Denormalize the input data using unit Gaussian normalization.
    """

    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = np.mean(x, 0)
        self.std = np.std(x, 0)
        self.eps = eps

    def encode(self, x):
        """
        Normalize the input data using unit Gaussian normalization.

        Args:
            x (numpy.ndarray): The input data to be normalized.

        Returns:
            numpy.ndarray: The normalized data.
        """
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        """
        Denormalize the input data using unit Gaussian normalization.

        Args:
            x (numpy.ndarray): The normalized data to be denormalized.

        Returns:
            numpy.ndarray: The denormalized data.
        """
        return (x * (self.std + self.eps)) + self.mean


def subsample_and_flatten(matrix, stride):
    """
    Subsamples a matrix and flattens it into a 1D array.
    The subsampling is done by extracting the first and last rows and columns,
    and then extracting the interior elements based on the stride.
    The extracted elements are then flattened into a 1D array.
    The function returns the indices of the extracted elements and the flattened array.
    """
    # matrix: a 3D numpy array of shape (N, rows, cols)
    # stride: an integer representing the stride

    # Get the dimensions of the input matrix
    N, rows, cols = matrix.shape

    # Create a list to store the indices of the elements to be extracted
    indices = []

    # Add indices for the first row (left to right)
    indices.extend((0, j) for j in range(cols))

    # Add indices for the last row (left to right)
    if rows > 1:
        indices.extend((rows - 1, j) for j in range(cols))

    # Add indices for the first column (top to bottom, excluding corners)
    if rows > 2:
        indices.extend((i, 0) for i in range(1, rows - 1))

    # Add indices for the last column (top to bottom, excluding corners)
    if rows > 2:
        indices.extend((i, cols - 1) for i in range(1, rows - 1))

    # print(indices)
    # Generate indices for the interior elements based on the stride
    counter = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if counter % stride == 0:
                indices.append((i, j))
                counter = 0
            counter += 1

    # sort the indices
    indices.sort(key=lambda x: (x[0], x[1]))

    # Extract the elements from the matrix using the sorted indices
    result = [matrix[:, i, j] for i, j in indices]

    return np.array(indices).astype('float32'), np.array(result).T.astype('float32')


def subsample(vector, stride):
    """
    Subsamples a vector and flattens it into a 1D array.
    The subsampling is done by extracting the first and last elements,
    and then extracting interior elements based on the stride.
    The function returns the *normalized* indices of the extracted elements 
    and the flattened array.

    Args:
        vector: a 2D numpy array of shape (N, L)
        stride: an integer representing the stride

    Returns:
        norm_indices: 1D numpy array of normalized indices in [0,1] (float32)
        result: 2D numpy array of shape (N, num_selected) with the subsampled values (float32)
    """
    if len(vector.shape) != 2:
        N, _, L = vector.shape
    else:
        N, L = vector.shape

    indices = []

    # Always take first element
    indices.append(0)

    # Always take last element
    if L > 1:
        indices.append(L - 1)

    # Subsample interior elements
    counter = 0
    for i in range(1, L - 1):
        if counter % stride == 0:
            indices.append(i)
            counter = 0
        counter += 1

    # Sort indices
    indices = sorted(set(indices))

    # Normalize indices to [0,1]
    norm_indices = np.array(indices, dtype="float32") / (L - 1)

    # Extract elements
    if len(vector.shape) != 2:
        result = [vector[:, :, i] for i in indices]
    else:
        result = [vector[:, i] for i in indices]

    return norm_indices, np.array(result, dtype="float32").T


def patch_coords(matrix):
    """
    Generate coordinates for each element in the matrix.

    Args:
        matrix (ndarray): Input matrix.

    Returns:
        ndarray: Array of coordinates for each element in the matrix.
    """
    N, rows, cols = matrix.shape

    nx = np.linspace(0,1,num=cols)
    ny = np.linspace(0,1,num=rows)

    coords = np.array(np.meshgrid(nx,ny))
    
    return coords


def reflect_pad(data, padding):
    """
    Apply reflection padding to a 4D tensor with custom padding on each side.

    Args:
        data (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        padding (tuple): A tuple (left, right, top, bottom) specifying the padding size on each side.

    Returns:
        torch.Tensor: Tensor with reflection padding applied.
    """
    if len(data.shape) != 4:
        raise ValueError("Only 4D tensors are supported (shape should be (batch, channels, height, width))")

    left, right, top, bottom = padding
    batch_size, channels, h, w = data.shape

    # Calculate new dimensions
    out_h = h + top + bottom
    out_w = w + left + right

    # Preallocate output tensor
    output = data.new_zeros((batch_size, channels, out_h, out_w))

    # Copy the central part
    output[:, :, top:top+h, left:left+w] = data

    # Reflect top and bottom
    if top > 0:
        output[:, :, 0:top, left:left+w] = data[:, :, 1:top+1, :].flip(2)
    if bottom > 0:
        output[:, :, top+h:, left:left+w] = data[:, :, -bottom-1:-1, :].flip(2)

    # Reflect left and right
    if left > 0:
        output[:, :, :, 0:left] = output[:, :, :, left+1:left*2+1].flip(-1)
    if right > 0:
        output[:, :, :, left+w:] = output[:, :, :, left+w-right-1:left+w-1].flip(-1)

    return output


class Smoothing2d:
    """
    Inverse(-Laplacian) smoother with selectable boundary conditions.

    mode = "periodic" : FFT-based spectral filter (periodic BCs)
    mode = "dirichlet": sine-transform-based spectral filter (zero Dirichlet BCs)

    Args:
        mode: "periodic" or "dirichlet"
        eps: small regularization term
        hx, hy: grid spacings (for discrete Laplacian eigenvalues in Dirichlet mode)
    """
    def __init__(self, mode="periodic", eps=1e-6, hx=1.0, hy=1.0):
        assert mode in ("periodic", "dirichlet")
        self.mode = mode
        self.eps = float(eps)
        self.hx = float(hx)
        self.hy = float(hy)
        self._cache = {}

    # ---------- PERIODIC (FFT) ----------
    def _periodic_kernel(self, H, W, device, dtype):
        key = ("per", H, W, device, dtype)
        if key in self._cache:
            return self._cache[key]
        kx = torch.fft.fftfreq(H, d=1.0).to(device=device, dtype=dtype)   # (H,)
        ky = torch.fft.rfftfreq(W, d=1.0).to(device=device, dtype=dtype)  # (W//2+1,)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        k2 = KX**2 + KY**2
        self._cache[key] = k2
        return k2

    def _apply_periodic(self, x):
        N, H, W = x.shape
        xhat = torch.fft.rfft2(x)
        k2 = self._periodic_kernel(H, W, x.device, x.dtype)
        filt = 1.0 / (4.0 * (np.pi**2) * k2 + self.eps)
        yhat = xhat * filt
        y = torch.fft.irfft2(yhat, s=(H, W))
        return y

    # ---------- DIRICHLET (DST via orthonormal sine basis) ----------
    def _sine_mats(self, H, W, device, dtype):
        key = ("sin", H, W, device, dtype)
        if key in self._cache:
            return self._cache[key]

        i = torch.arange(1, H+1, device=device, dtype=dtype).view(H, 1)
        j = torch.arange(1, H+1, device=device, dtype=dtype).view(1, H)
        k = torch.arange(1, W+1, device=device, dtype=dtype).view(W, 1)
        l = torch.arange(1, W+1, device=device, dtype=dtype).view(1, W)

        Sx = torch.sin(np.pi * i * j / (H + 1)) * torch.sqrt(2.0 / torch.tensor(H+1, dtype=dtype, device=device))
        Sy = torch.sin(np.pi * k * l / (W + 1)) * torch.sqrt(2.0 / torch.tensor(W+1, dtype=dtype, device=device))
        self._cache[key] = (Sx, Sy)
        return Sx, Sy

    def _dirichlet_eigs(self, H, W, device, dtype):
        key = ("lam", H, W, device, dtype, self.hx, self.hy)
        if key in self._cache:
            return self._cache[key]
        p = torch.arange(1, H+1, device=device, dtype=dtype).view(H, 1)
        q = torch.arange(1, W+1, device=device, dtype=dtype).view(1, W)
        lam_x = 4.0 * torch.sin(np.pi * p / (2.0 * (H + 1)))**2 / (self.hx**2)
        lam_y = 4.0 * torch.sin(np.pi * q / (2.0 * (W + 1)))**2 / (self.hy**2)
        lam = lam_x + lam_y
        self._cache[key] = lam
        return lam

    def _apply_dirichlet(self, x):
        N, H, W = x.shape
        device, dtype = x.device, x.dtype
        Sx, Sy = self._sine_mats(H, W, device, dtype)
        lam = self._dirichlet_eigs(H, W, device, dtype)

        # DST: û = Sx^T U Sy
        Ut = torch.matmul(Sx.T, x)     # (H,H) @ (N,H,W)  -> broadcasts
        U_hat = torch.matmul(Ut, Sy)   # (N,H,W)

        # Filter
        U_hat = U_hat / (lam + self.eps)

        # Inverse DST: U = Sx û Sy^T
        Ut2 = torch.matmul(Sx, U_hat)
        y = torch.matmul(Ut2, Sy.T)
        return y

    def __call__(self, x):
        """
        x: (N, H, W) tensor
        """
        assert x.dim() == 3, "Expected x with shape (N, H, W)"
        if self.mode == "periodic":
            return self._apply_periodic(x)
        else:
            return self._apply_dirichlet(x)