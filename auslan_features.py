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

    if hand_landmarks_list:
        for i, lm in enumerate(hand_landmarks_list):
            handed = handedness_list[i] if handedness_list and i < len(handedness_list) else []
            if handed:
                label = handed[0].category_name
            else:
                label = "Left" if lm[0].x < 0.5 else "Right"
            pts = _hand_xyz(lm)
            normed = _normalize_hand(pts)
            # MediaPipe sometimes labels both hands the same. Keep both by
            # filling whichever slot is still empty.
            if label == "Left" and left_pts is None:
                left = normed
                left_pts = pts
            elif label != "Left" and right_pts is None:
                right = normed
                right_pts = pts
            elif left_pts is None:
                left = normed
                left_pts = pts
            elif right_pts is None:
                right = normed
                right_pts = pts

    if left_pts is not None and right_pts is not None:
        rel = _rel_features(left_pts, right_pts)
    else:
        rel = np.zeros(REL_DIM, dtype=np.float32)

    return np.concatenate([left, right, rel]).astype(np.float32)


# Two-handed vowels point at one finger of the base hand.
VOWEL_TIPS = {"A": 4, "E": 8, "I": 12, "O": 16, "U": 20}


def index_contact(hand_landmarks_list):
    """Where the pointing index sits on the other hand, in the camera image.

    Auslan T is that tip in the middle of the palm. The vowels are the same
    point on a fingertip: A thumb, E index, I middle, O ring, U little finger.
    Image x,y only: each hand's z is relative to its own wrist, so it cannot
    be compared across hands. Returns the letter, or None.
    """
    if not hand_landmarks_list or len(hand_landmarks_list) < 2:
        return None
    hands = [
        np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        for lm in hand_landmarks_list[:2]
    ]

    def extension(pts, tip, mcp):
        return float(np.linalg.norm(pts[tip] - pts[0]) / (np.linalg.norm(pts[mcp] - pts[0]) + 1e-6))

    def flat(pts):
        return pts[:, :2]

    def pointer_score(pts):
        index = extension(pts, 8, 5)
        folded = [extension(pts, tip, mcp) for tip, mcp in ((12, 9), (16, 13), (20, 17))]
        if index < 1.45 or folded[0] > 1.2:
            return -1.0
        return index - max(folded)

    best = None
    for i, pointer in enumerate(hands):
        base = hands[1 - i]
        open_fingers = sum(
            extension(base, tip, mcp) > 1.45
            for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17))
        )
        if open_fingers < 3:
            continue
        # Middle of the palm, between the wrist and the middle knuckle.
        # Distances stay in image x,y. z is not shared between the two hands.
        base_xy = flat(base)
        pointer_xy = flat(pointer)
        palm = 0.45 * base_xy[0] + 0.55 * base_xy[9]
        tip_xy = pointer_xy[8]
        d_palm = float(np.linalg.norm(tip_xy - palm))
        d_knuckle = float(np.linalg.norm(pointer_xy[5] - palm))
        index_ext = extension(pointer, 8, 5)
        middle_ext = extension(pointer, 12, 9)
        candidate = None
        # Lower-case r lies on the palm: index hooked, the other fingers only
        # partly closed, knuckle on the palm as well. A straight point is T.
        if (
            1.25 <= index_ext <= 1.72
            and 1.02 <= middle_ext <= 1.55
            and d_palm <= 0.14
            and d_knuckle <= 0.16
        ):
            candidate = ("R", d_palm)
        elif pointer_score(pointer) >= 0.35:
            dists = {
                name: float(np.linalg.norm(tip_xy - base_xy[idx]))
                for name, idx in VOWEL_TIPS.items()
            }
            finger_name, d_finger = min(dists.items(), key=lambda item: item[1])
            nxt = sorted(dists.values())[1]
            # Tip in the palm, and the pointing knuckle still off the palm.
            # A finger laid flat (L) keeps that knuckle on the palm too.
            if d_palm <= 0.16 and d_palm < d_finger * 0.8 and d_knuckle > d_palm * 1.5:
                candidate = ("T", d_palm)
            elif d_finger <= 0.07 and d_finger < d_palm * 0.8 and d_finger <= nxt * 0.7:
                candidate = (finger_name, d_finger)
        if candidate and (best is None or candidate[1] < best[1]):
            best = candidate
    return None if best is None else best[0]


def finger_split(hand_landmarks_list):
    """V is index and middle spread apart. N keeps that pair together.

    Measured on the recorded hands, in wrist-centred units: V is about 0.52
    apart, N about 0.22. Ring and little finger stay folded for both.
    """
    if not hand_landmarks_list:
        return None
    for lm in hand_landmarks_list[:2]:
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        pts = pts - pts[0]
        scale = float(np.linalg.norm(pts[9])) + 1e-6

        def extension(tip, mcp):
            return float(np.linalg.norm(pts[tip]) / (np.linalg.norm(pts[mcp]) + 1e-6))

        if extension(8, 5) < 1.7 or extension(12, 9) < 1.85:
            continue
        if extension(16, 13) > 1.28 or extension(20, 17) > 1.15:
            continue
        spread = float(np.linalg.norm(pts[8] - pts[12]) / scale)
        if spread >= 0.40:
            return "V"
    return None


def pointed_vowel(hand_landmarks_list):
    """Which vowel finger one index tip is touching, in camera coordinates.

    A is the thumb, E the index, I the middle, O the ring, U the little finger.
    Returns (letter, nearest_distance, next_distance) or None when it is not clear.
    """
    if not hand_landmarks_list or len(hand_landmarks_list) < 2:
        return None
    hands = []
    for lm in hand_landmarks_list[:2]:
        hands.append(np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32))

    def scores(pointer, base):
        tip = pointer[8]
        return {name: float(np.linalg.norm(tip - base[idx])) for name, idx in VOWEL_TIPS.items()}

    def best(score):
        ordered = sorted(score.items(), key=lambda item: item[1])
        return ordered[0][0], ordered[0][1], ordered[1][1]

    first = best(scores(hands[0], hands[1]))
    second = best(scores(hands[1], hands[0]))
    letter, nearest, nxt = first if first[1] <= second[1] else second
    # Must be on that fingertip, and clearly nearer than the next finger.
    # A touch is a small fraction of the image; a hand beside the fingers is not.
    if nxt <= 1e-6 or nearest > 0.05 or nearest > nxt * 0.65:
        return None
    return letter, nearest, nxt


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
