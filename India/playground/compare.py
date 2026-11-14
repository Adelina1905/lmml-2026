import cv2, numpy as np, sys

def prepare(path, size=500):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # 1. colour doesn't matter -> grey
    h, w = gray.shape
    s = size / max(h, w)
    gray = cv2.resize(gray, (int(w*s), int(h*s)))         # 2. same size for everyone
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)                              # 3. equalise contrast

def compare(path_a, path_b):
    a, b = prepare(path_a), prepare(path_b)
    orb = cv2.ORB_create(nfeatures=1500)                  # 4. find corners + describe them
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None: return 0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)                  # 5. match descriptors
    good = []
    for pair in bf.knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 4: return len(good), None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    # 6. ONE rotation+scale+shift must explain the matches
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=5)
    if M is None: return 0, None
    scale = float(np.hypot(M[0,0], M[1,0])); angle = float(np.degrees(np.arctan2(M[1,0], M[0,0])))
    return int(inl.sum()), (round(scale,3), round(angle,2), len(good))

if __name__ == "__main__":
    print(compare(sys.argv[1], sys.argv[2]))
