import os, sys, csv, zipfile, tempfile, cv2, numpy as np

THRESH, MIN_RATIO, HASH_MAX, K = 180, 0.0, 4, 30
# THRESH -minimum RANSAC inliners
# MIN_RATIO - minimum inlier ration
# HASH_MAX - maximum accepted Hamming distance
# K - number of nearest hash neighbors   


# unpack the input
src = sys.argv[1] if len(sys.argv) > 1 else "test-input.zip"  # Unpacking the input 
if os.path.isfile(src) and zipfile.is_zipfile(src): # Looks for the situations, first, input is a .zip
    work = tempfile.mkdtemp(); zipfile.ZipFile(src).extractall(work)
else: # input is a dir
    work = src

# collect images
paths = {} # stores all their names and paths 
for root, _, fs in os.walk(work):
    for f in fs:
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._"):
            paths[f] = os.path.join(root, f)
files = sorted(paths) # creates an alphabetic sorted 
idx = {f: i for i, f in enumerate(files)} # creates a mapping between an image and its numeric position (later its needed for storing in a numpy matrix)

# fingerprints
# Implement pHash manually using OpenCV instead of a library
def phash(path):  
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # loads the image in grayscale, its the process the process of converting a multi-channel color image (such as RGB or BGR) into a single-channel image representing only light intensity (shades of gray)
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) # every image is resized to a 32x32 pixels, then converts the pixels values into 32-bit floating numbers
    low = cv2.dct(small)[:8, :8] # The DCT transforms an image from the spatial domain into the frequency domain.
    return (low > np.median(low)).flatten() # convert coefficients into numerical values
bits = np.array([phash(paths[f]) for f in files], dtype=np.uint8) # calculate hashes for all images, the result is a numpy matrix

# shortlist: K closest fingerprints per image (The goal is to find the 10 most visually similar images for each image.) 
candidates = set()
for i, a in enumerate(files):
    d = (bits != bits[i]).sum(axis=1) # compares the fingerprints of image i with every fingertip in the matrix (sum counts the number of different bits for every image)
    d[i] = 999 # exclude the image itself
    for j in np.argsort(d)[:K]: # select K closest images
        candidates.add(tuple(sorted((a, files[int(j)])))) # store unique pairs
print(len(files), "images,", len(candidates), "candidates")

# keypoint check
#this section the detailed image-matching mechanism

sift = cv2.SIFT_create(nfeatures=1000) # sift - stands for scale-invariant feature transform it detects corners, textures, building edges, distinctive object patterns
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)) # flann - stands for fast library for approximate nearest neighbors (its role is to find similar )
feats = {}


def get(f):
    if f not in feats:
        img = cv2.imread(paths[f], cv2.IMREAD_GRAYSCALE)
        s = 800 / max(img.shape)
        if s < 1: img = cv2.resize(img, None, fx=s, fy=s)
        feats[f] = sift.detectAndCompute(img, None)
    return feats[f]

def agreeing_points(a, b):
    ka, da = get(a); kb, db = get(b)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10: return 0, 0.0
    matches = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10: return 0, 0.0
    src_pts = np.float32([ka[m.queryIdx].pt for m in good]); dst_pts = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return (0, 0.0) if mask is None else (int(mask.sum()), float(mask.mean()))

# score, decide, write
scores = []
for n, (a, b) in enumerate(sorted(candidates)):
    inl, ratio = agreeing_points(a, b)
    scores.append((a, b, inl, ratio, int((bits[idx[a]] != bits[idx[b]]).sum())))
    if n % 500 == 0: print(n, "/", len(candidates))

with open("output.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image_a", "image_b"])
    for a, b, inl, ratio, hd in scores:
        if (inl >= THRESH and ratio >= MIN_RATIO) or hd <= HASH_MAX:
            w.writerow([a, b])

if os.environ.get("DEBUG"):
    with open("all_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image_a", "image_b", "inliers", "ratio", "hash_dist"])
        w.writerows(scores)
print("done")