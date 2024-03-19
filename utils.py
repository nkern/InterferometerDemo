import numpy as np
import cv2


def rotation_mat(phi):
	return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


class UVSample:
	def __init__(self, ants, rotate=None, Nu=128, ratio=3.0/4.0, keep_zero=True):
		"""
		Args:
			ants : ndarray of shape (2, Nants)
				Antenna positions in XY space.
				Maximum separation for X-values of antennas should be 1.
			rotate : float
				Rotation of uv-plane in degrees, mimicking earth rotation synthesis.
			Nu : int
				Number of samples along the U direction of uv plane
			ratio : float
				Aspect ratio of uv plane: Nv = Nu * ratio
			keep_zero : bool
				If True, keep u=v=0 sampling in mask
		"""
		self.Nu = Nu
		self.Nv = int(Nu * ratio)
		self.shape = (self.Nv, self.Nu)
		self.keep_zero = keep_zero

		self.compute_mask(ants, rotate=rotate)

	def compute_mask(self, ants, rotate=None):
		# compute uv sampling from antenna array
		self.ants = ants
		uvs = (ants[:, None, :] - ants[:, :, None]).reshape(2, -1)

		# only keep u >= 0 values
		uvs = uvs[:, uvs[0] > 0]

		# normalize uvs to this aspect ratio
		# TODO: if using vertical aspect ratio, this should be uvs * self.Nu // 2
		self.uvs = uvs * (self.Nv // 2)

		if rotate is not None and rotate >= 5.0:
			rotations = np.arange(0, rotate+.1, 5.0)
			_uvs = []
			for rot in rotations:
				_uvs.append(rotation_mat(rot * np.pi / 180) @ self.uvs)
			uvs = np.concatenate(_uvs, axis=-1)
			self.uvs = uvs[:, uvs[0] > 0]

		# construct mask
		self.mask = np.zeros((self.Nv, self.Nu//2+1, 1), dtype=np.float32)

		x_idx = np.digitize(self.uvs[0], np.arange(self.Nu//2)+.5, right=False)
		y_idx = np.digitize(self.uvs[1] + self.Nv//2, np.arange(self.Nv-1)+.5, right=False)

		self.mask[y_idx, x_idx] = 1.0

		if self.keep_zero:
			self.mask[self.Nv//2, 0] = 1.0

		# normalize by mask filling factor
		# this sets PSF to peak-normalized
		# meaning if there is diffuse structure, then the reconstructed
		# image will have max values > 255 and needs re-normalization
		self.mask *= (self.mask.size / self.mask.sum())


	def rfft(self, im):
		return np.fft.fftshift(np.fft.rfft2(im, axes=(0, 1)), axes=(0,))

	def fft_convolve(self, im):
		imfft = self.rfft(im)
		return np.fft.irfft2(np.fft.ifftshift(imfft * self.mask, axes=(0,)), s=im.shape[:2], axes=(0, 1)), imfft

	def __call__(self, im):
		# fft, multiply by mask, and ifft
		imc, imfft = self.fft_convolve(im)
		# normalize by imc.max()
		imc *= 255 / imc.max()
		return imc.clip(0, 255).astype(np.uint8), imfft


class Image:
	def __init__(self, img=None, shape=None):
		self.shape = shape
		self.store(img)

	def store(self, img=None):
		if img is not None:
			if self.shape is not None and img.shape != self.shape:
				img = cv2.resize(img, self.shape[::-1], cv2.INTER_NEAREST_EXACT)
			if img.ndim != 3:
				img = img[..., None]
			self.img = img


def make_hex(N, D=.1):
    x, y, ants = [], [], []
    ant = 0
    k = 0
    start = 0
    for i in range(2*N - 1):
        for j in range(N + k):
            x.append(j + start)
            y.append(i * np.sin(np.pi/3))
            ants.append(ant)
            ant += 1
        if i < N-1:
            k += 1
            start -= .5
        else:
            k -= 1
            start += .5
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    return np.vstack([x, y]) * D

