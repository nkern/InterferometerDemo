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

		# only keep the u > 0 half: (-u, -v) is the conjugate of (u, v) for a real sky
		uvs = uvs[:, uvs[0] > 0]

		# scale uvs so u and v share units (1 unit = 1 / Nv cycles per image pixel);
		# keeping them isotropic here makes the rotation below a true rotation
		self.uvs = uvs * (self.Nv // 2)

		if rotate is not None and rotate >= 5.0:
			rotations = np.arange(0, rotate+.1, 5.0)
			_uvs = []
			for rot in rotations:
				_uvs.append(rotation_mat(rot * np.pi / 180) @ self.uvs)
			uvs = np.concatenate(_uvs, axis=-1)
			# baselines rotated past u = 0 are kept as their conjugate (-u, -v), since
			# the conjugate baseline was already dropped before rotation
			self.uvs = uvs * np.where(uvs[0] < 0, -1, 1)

		# construct mask
		self.mask = np.zeros((self.Nv, self.Nu//2+1, 1), dtype=np.float32)

		# u is sampled across Nu image columns but v across Nv rows, so convert u to
		# grid cells (u * Nu / Nv, i.e. baseline * Nu // 2) to keep l and m pixels the same size
		x_idx = np.digitize(self.uvs[0] * self.Nu / self.Nv, np.arange(self.Nu//2)+.5, right=False)
		# image rows count downward, so map -v to rows to keep +v (and +m) pointing up
		y_idx = np.digitize(-self.uvs[1] + self.Nv//2, np.arange(self.Nv-1)+.5, right=False)

		self.mask[y_idx, x_idx] = 1.0

		if self.keep_zero:
			self.mask[self.Nv//2, 0] = 1.0

		# normalize by mask filling factor
		# this sets PSF to peak-normalized
		# meaning if there is diffuse structure, then the reconstructed
		# image will have max values > 255 and needs re-normalization
		self.mask *= (self.mask.size / self.mask.sum())


	def psf(self):
		# PSF is the inverse FFT of the uv sampling mask: centered and peak-normalized
		psf = np.fft.irfft2(np.fft.ifftshift(self.mask[..., 0], axes=(0,)), s=self.shape)
		psf = np.fft.fftshift(psf)
		return psf / psf.max()

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


def make_spiral(N, Narms=4, spacing=.035, b=.05):
    """
    N antennas per arm on Narms Archimedean spiral arms (r = b * theta),
    placed at a fixed arc-length spacing, so increasing N extends each arm
    without moving the inner antennas.
    """
    # invert the arc length of r = b * theta to find each antenna's theta
    theta = np.linspace(0, 4*np.pi, 4001)
    arc = b / 2 * (theta * np.sqrt(1 + theta**2) + np.arcsinh(theta))
    theta_ants = np.interp(spacing * np.arange(1, N + 1), arc, theta)
    ants = []
    for k in range(Narms):
        phi = theta_ants + 2 * np.pi * k / Narms
        ants.append(np.vstack([b * theta_ants * np.cos(phi), b * theta_ants * np.sin(phi)]))
    return np.concatenate(ants, axis=1)

