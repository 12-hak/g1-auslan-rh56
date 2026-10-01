import numpy as np


LM_PER_HAND = 21
COORDS = 3
HAND_DIM = LM_PER_HAND * COORDS
REL_DIM = 12
FEATURE_DIM = HAND_DIM * 2 + REL_DIM


def _hand_xyz(landmarks):
    pts = np.zeros((LM_PER_HAND, COORDS), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x
        pts[i, 1] = lm.y
        pts[i, 2] = lm.z
    return pts


def _normalize_hand(pts):
    out = pts.copy()
    wrist = out[0].copy()
    out -= wrist
    scale = np.linalg.norm(out[9]) + 1e-6
    out /= scale
    return out.reshape(-1)


def _rel_features(left_pts, right_pts):
    lw = left_pts[0]
    rw = right_pts[0]
    delta = rw - lw
    dist = np.linalg.norm(delta) + 1e-6
    l_idx = left_pts[8] - lw
    r_idx = right_pts[8] - rw
    l_th = left_pts[4] - lw
    r_th = right_pts[4] - rw
    return np.array(
        [
            delta[0] / dist,
            delta[1] / dist,
            delta[2] / dist,
            dist,
            np.linalg.norm(l_idx),
            np.linalg.norm(r_idx),
            np.linalg.norm(l_th),
            np.linalg.norm(r_th),
            np.dot(l_idx, r_idx) / (np.linalg.norm(l_idx) * np.linalg.norm(r_idx) + 1e-6),
            np.dot(l_th, r_th) / (np.linalg.norm(l_th) * np.linalg.norm(r_th) + 1e-6),
            np.linalg.norm(left_pts[8] - right_pts[8]),
            np.linalg.norm(left_pts[4] - right_pts[4]),
        ],
        dtype=np.float32,
    )


def extract_features(hand_landmarks_list, handedness_list):
    """
    Build a fixed-size feature vector from up to 2 MediaPipe hands.
    Order in vector: Left hand, Right hand, relative. Missing hands are zeros.
    """
    left = np.zeros(HAND_DIM, dtype=np.float32)
    right = np.zeros(HAND_DIM, dtype=np.float32)
    left_pts = None
    right_pts = None

    if hand_landmarks_list and handedness_list:
        for lm, handed in zip(hand_landmarks_list, handedness_list):
            label = handed[0].category_name
            pts = _hand_xyz(lm)
            normed = _normalize_hand(pts)
            if label == "Left":
                left = normed
                left_pts = pts
            else:
                right = normed
                right_pts = pts

    if left_pts is not None and right_pts is not None:
        rel = _rel_features(left_pts, right_pts)
    else:
        rel = np.zeros(REL_DIM, dtype=np.float32)

    return np.concatenate([left, right, rel]).astype(np.float32)


def finger_curl_ratios(landmarks):
    """Rough per-finger extension ratios from a single hand (for rule fallback / poses)."""
    pts = _hand_xyz(landmarks)
    wrist = pts[0]

    def ratio(tip, mcp):
        d_tip = np.linalg.norm(pts[tip] - wrist)
        d_mcp = np.linalg.norm(pts[mcp] - wrist) + 1e-6
        return float(d_tip / d_mcp)

    return {
        "pinky": ratio(20, 17),
        "ring": ratio(16, 13),
        "middle": ratio(12, 9),
        "index": ratio(8, 5),
        "thumb": ratio(4, 2),
    }
