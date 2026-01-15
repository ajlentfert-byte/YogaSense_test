# pose.py
import math

# ------------------------
# Helpers
# ------------------------

def joint_angle_2d(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 180
    cosang = max(-1, min(1, (ba[0]*bc[0] + ba[1]*bc[1]) / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosang))


def score_angle(actual, target, tolerance):
    return max(0, 1 - abs(actual - target) / tolerance)


def torso_length(lm):
    ls, rs = lm[11], lm[12]
    lh, rh = lm[23], lm[24]
    shoulder_mid = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    hip_mid = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
    return math.dist(shoulder_mid, hip_mid)


# ------------------------
# MOUNTAIN
# ------------------------

def mountain_pose_score(lm):
    ls, rs = lm[11], lm[12]
    lw, rw = lm[15], lm[16]
    lh, lk, la = lm[23], lm[25], lm[27]
    rh, rk, ra = lm[24], lm[26], lm[28]
    le, re = lm[13], lm[14]

    shoulder_width = abs(ls.x - rs.x)
    wrist_width = abs(lw.x - rw.x)

    # --- arm height ---
    left_height = max(0, min(1, (ls.y - lw.y) / 0.25))
    right_height = max(0, min(1, (rs.y - rw.y) / 0.25))

    # --- arm straightness ---
    left_straight = score_angle(joint_angle_2d(ls, le, lw), 180, 60)
    right_straight = score_angle(joint_angle_2d(rs, re, rw), 180, 60)

    # --- arm width (the new part) ---
    # ideal: wrists slightly wider than shoulders
    width_ratio = wrist_width / shoulder_width

    # Ideal relaxed V is around 2 shoulder widths
    ideal = 1.8
    tolerance = 0.8  # how relaxed we are about it

    width_score = max(
        0,
        1 - abs(width_ratio - ideal) / tolerance
    )

    scores = {
        "left_arm": 0.7 * left_height + 0.3 * left_straight,
        "right_arm": 0.7 * right_height + 0.3 * right_straight,
        "arm_width": width_score,
        "left_leg": score_angle(joint_angle_2d(lh, lk, la), 180, 35),
        "right_leg": score_angle(joint_angle_2d(rh, rk, ra), 180, 35),
    }

    accuracy = sum(scores.values()) / len(scores) * 100
    return accuracy, None, scores


# ------------------------
# TREE
# ------------------------

def tree_pose_score(lm):
    ls, rs = lm[11], lm[12]
    lw, rw = lm[15], lm[16]
    lh, lk, la = lm[23], lm[25], lm[27]
    rh, rk, ra = lm[24], lm[26], lm[28]

    torso = torso_length(lm)



    left_leg = joint_angle_2d(lh, lk, la)
    right_leg = joint_angle_2d(rh, rk, ra)
    standing_leg = left_leg if abs(left_leg - 180) < abs(right_leg - 180) else right_leg
    standing_leg_score = score_angle(standing_leg, 180, 30)

    left_lift = abs(la.y - lk.y) / torso
    right_lift = abs(ra.y - rk.y) / torso
    lifted = min(left_lift, right_lift)

    if lifted < 0.15:
        lifted_score = 1.0
    elif lifted > 0.35:
        lifted_score = 0.0
    else:
        lifted_score = 1 - (lifted - 0.15) / 0.2

    wrist_dist = math.dist((lw.x, lw.y), (rw.x, rw.y)) / torso
    hands_score = max(0, 1 - wrist_dist * 2)

    avg_wrist_y = (lw.y + rw.y) / 2
    avg_shoulder_y = (ls.y + rs.y) / 2
    height_score = max(0, 1 - abs(avg_wrist_y - avg_shoulder_y) / 0.35)

    scores = {
        "standing_leg": standing_leg_score,
        "lifted_foot": lifted_score,
        "hands": hands_score,
        "hand_height": height_score
    }

    accuracy = sum(scores.values()) / len(scores) * 100
    return accuracy, None, scores


# ------------------------
# WARRIOR II
# ------------------------

def warrior2_pose_score(lm):
    ls, rs = lm[11], lm[12]
    le, re = lm[13], lm[14]
    lw, rw = lm[15], lm[16]
    lh, lk, la = lm[23], lm[25], lm[27]
    rh, rk, ra = lm[24], lm[26], lm[28]

    left_knee = joint_angle_2d(lh, lk, la)
    right_knee = joint_angle_2d(rh, rk, ra)

    if abs(left_knee - 95) < abs(right_knee - 95):
        front_knee, back_knee = left_knee, right_knee
    else:
        front_knee, back_knee = right_knee, left_knee

    front_score = score_angle(front_knee, 105, 90)
    back_score = score_angle(back_knee, 180, 70)

    left_arm = max(0, 1 - abs(ls.y - lw.y) / 0.3)
    right_arm = max(0, 1 - abs(rs.y - rw.y) / 0.3)

    def elbow_score(s, e, w):
        angle = joint_angle_2d(s, e, w)
        angle_part = max(0, min(1, (angle - 135) / 45))
        height_part = max(0, 1 - abs(s.y - w.y) / 0.3)
        return (angle_part + height_part) / 2

    scores = {
        "front_knee": front_score,
        "back_knee": back_score,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_elbow": elbow_score(ls, le, lw),
        "right_elbow": elbow_score(rs, re, rw)
    }

    accuracy = sum(scores.values()) / len(scores) * 100
    return accuracy, None, scores
