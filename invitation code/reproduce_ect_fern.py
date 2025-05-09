
import itertools
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import scipy.special as special

def neighborhood_setup(dimension, downsample=1):
    neighs = sorted(list(itertools.product(range(2), repeat=dimension)), key=np.sum)[1:]
    neighs = list(map(tuple, np.array(neighs) * downsample))
    subtuples = dict()
    for i in range(len(neighs)):
        subtup = [0]
        for j in range(len(neighs)):
            if np.all(np.subtract(neighs[i], neighs[j]) > -1):
                subtup.append(j + 1)
        subtuples[neighs[i]] = subtup
    return neighs, subtuples

def neighborhood(voxel, neighs, hood, dcoords):
    hood[0] = dcoords[voxel]
    neighbors = np.add(voxel, neighs)
    for j in range(1, len(hood)):
        key = tuple(neighbors[j - 1, :])
        if key in dcoords:
            hood[j] = dcoords[key]
    return hood

def centerVertices(verts):
    origin = -1 * np.mean(verts, axis=0)
    return np.add(verts, origin)

class CubicalComplex:
    def __init__(self, img):
        self.img = img

    def complexify(self, center=True, downsample=1):
        scoords = np.nonzero(self.img)
        scoords = np.vstack(scoords).T
        skip = np.zeros(self.img.ndim, dtype=int) + downsample
        coords = scoords[np.all(np.fmod(scoords, skip) == 0, axis=1), :]
        keys = [tuple(coords[i, :]) for i in range(len(coords))]
        dcoords = dict(zip(keys, range(len(coords))))
        neighs, subtuples = neighborhood_setup(self.img.ndim, downsample)
        binom = [special.comb(self.img.ndim, k, exact=True) for k in range(self.img.ndim + 1)]
        hood = np.zeros(len(neighs) + 1, dtype=np.int32) - 1
        cells = [[] for _ in range(self.img.ndim + 1)]

        for voxel in dcoords:
            hood.fill(-1)
            hood = neighborhood(voxel, neighs, hood, dcoords)
            nhood = hood > -1
            c = 0
            if np.all(nhood[:-1]):
                for k in range(1, self.img.ndim):
                    for _ in range(binom[k]):
                        cell = hood[subtuples[neighs[c]]]
                        cells[k].append(cell)
                        c += 1
                if nhood[-1]:
                    cells[self.img.ndim].append(hood.copy())
            else:
                for k in range(1, self.img.ndim):
                    for _ in range(binom[k]):
                        cell = nhood[subtuples[neighs[c]]]
                        if np.all(cell):
                            cells[k].append(hood[subtuples[neighs[c]]])
                        c += 1

        dim = self.img.ndim
        for k in range(dim, -1, -1):
            if len(cells[k]) > 0:
                break

        self.ndim = dim
        self.cells = [np.array(cells[k]) for k in range(dim + 1)]
        self.cells[0] = centerVertices(coords) if center else coords
        return self

    def ECC(self, filtration, T=32, bbox=None):
        if bbox is None:
            minh, maxh = np.min(filtration), np.max(filtration)
        else:
            minh, maxh = bbox
        buckets = [None for _ in range(len(self.cells))]
        buckets[0], _ = np.histogram(filtration, bins=T, range=(minh, maxh))
        for i in range(1, len(buckets)):
            if len(self.cells[i]) > 0:
                buckets[i], _ = np.histogram(np.max(filtration[self.cells[i]], axis=1), bins=T, range=(minh, maxh))
        ecc = np.zeros_like(buckets[0])
        for i in range(len(buckets)):
            if buckets[i] is not None:
                ecc = np.add(ecc, ((-1) ** i) * buckets[i])
        return np.cumsum(ecc)

    def ECT(self, directions, T=32, verts=None, bbox=None):
        if verts is None:
            verts = self.cells[0]
        ect = np.zeros(T * directions.shape[0], dtype=int)
        for i in range(directions.shape[0]):
            heights = np.sum(verts * directions[i, :], axis=1)
            ecc = self.ECC(heights, T, bbox)
            ect[i * T: (i + 1) * T] = ecc
        return ect

def regular_directions(N=50, r=1, dims=2):
    if dims == 2:
        eq_angles = np.linspace(0, 2 * np.pi, num=N, endpoint=False)
        return np.column_stack((np.cos(eq_angles), np.sin(eq_angles)))
    else:
        raise ValueError("Only 2D directions are supported.")

# --- Main Execution ---
image = tf.imread("leafbw.tif")
cube = CubicalComplex(image).complexify(center=True)
directions = regular_directions(N=64, dims=2)
T = 100
ect_matrix = cube.ECT(directions, T=T).reshape(len(directions), T).T

plt.figure(figsize=(12, 6))
plt.imshow(ect_matrix, cmap='viridis', aspect='auto', origin='lower')
plt.colorbar(label='Euler Characteristic')
plt.xlabel("Direction Index")
plt.ylabel("Threshold Index")
plt.title("ECT Matrix of Fern Leaf")
plt.tight_layout()
plt.savefig("ect_matrix_fern.png", dpi=300)
plt.savefig("ect_matrix_fern.jpg", dpi=300)
plt.close()
print("Saved ECT matrix to 'ect_matrix_fern.png' and 'ect_matrix_fern.jpg'")
