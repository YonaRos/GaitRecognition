import numpy as np
import torch
from torch import tensor
import cv2
import r_tensor_ff
import pandas as pd


vid_dic = {}

tensors_array0, reg0 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f1d5_1_regular_mmpose')
tensors_array1, reg1 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f1d10_1_regular_mmpose')
tensors_array2, reg2 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f1d15_1_regular_mmpose')
tensors_array3, reg3 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f1d35_1_regular_mmpose')
tensors_array4, reg4 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f2d20_1_regular_mmpose')
tensors_array5, reg5 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f2d40_1_regular_mmpose')
tensors_array6, reg6 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f2d50_1_regular_mmpose')
tensors_array7, reg7 = r_tensor_ff.read_func(
    'C:\\Users\\ofirs\OneDrive\\שולחן העבודה\\סמסטר ח\\final project\\codes\\tensors_directory\\regular_mmpose\\f2d75_1_regular_mmpose')

for i in range(8):
    tensor_name = "tensors_array" + str(i)
    vid_dic[tensor_name] = eval(tensor_name)  # Using eval() to evaluate the variable by its name

def write_to_excel(data):
    df = pd.DataFrame([data])
    df.T.to_excel('array_sums.xlsx', index=False, header=False)

def length_between_two_points(point1, point2):
    return torch.norm(point2 - point1)

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norm_product
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def z_angle(skeleton):
    bottom_avg = skeleton[[3, 6], :].mean(dim=0)
    head_avg = skeleton[[0, 7, 8], :].mean(dim=0)
    vector = head_avg - bottom_avg
    x, y, z = vector
    angle = np.rad2deg(np.arctan(np.sqrt(x ** 2 + y ** 2) / z))
    return angle


def angle_with_xy(p1, p2):
    vector = p2 - p1
    vx, vy, vz = vector
    vx, vy, vz = vx.item(), vy.item(), vz.item()
    r = np.sqrt(vx ** 2 + vy ** 2)
    angle = np.arctan(vz / r)
    angle_degrees = np.degrees(angle)
    return angle_degrees


def angle_with_z(p1, p2):
    vector = p2 - p1
    vx, vy, vz = vector
    r = np.sqrt(vx ** 2 + vy ** 2)
    angle = np.arctan(r / vz)
    angle_degrees = np.degrees(angle)
    return angle_degrees


def legs_length_difference(skeleton):
    left_leg_length_down = torch.norm(skeleton[[2], :] - skeleton[[3], :])
    left_leg_length_up = torch.norm(skeleton[[1], :] - skeleton[[2], :])
    left_leg_len = left_leg_length_down + left_leg_length_up
    right_leg_length_down = torch.norm(skeleton[[5], :] - skeleton[[6], :])
    right_leg_length_up = torch.norm(skeleton[[4], :] - skeleton[[5], :])
    right_leg_len = right_leg_length_down + right_leg_length_up
    return abs(right_leg_len - left_leg_len)


def hands_and_shoulders_length_difference(skeleton):
    left_hand_length_down = torch.norm(skeleton[[15], :] - skeleton[[16], :])
    left_hand_length_middle = torch.norm(skeleton[[14], :] - skeleton[[15], :])
    shoulder_to__left_hand = torch.norm(skeleton[[9], :] - skeleton[[14], :])
    left_hand_len = left_hand_length_down + left_hand_length_middle + shoulder_to__left_hand
    right_hand_length_down = torch.norm(skeleton[[12], :] - skeleton[[13], :])
    right_hand_length_middle = torch.norm(skeleton[[11], :] - skeleton[[12], :])
    shoulder_to__right_hand = torch.norm(skeleton[[9], :] - skeleton[[8], :])
    right_hand_len = right_hand_length_down + right_hand_length_middle + shoulder_to__right_hand
    return abs(left_hand_len - right_hand_len)


def human_height(hip, r_hip, r_knee, r_foot, l_hip, l_knee, l_foot, belly, neck, head):
    left_leg = length_between_two_points(l_knee, l_foot) + length_between_two_points(l_hip, l_knee)
    right_leg = length_between_two_points(r_knee, r_foot) + length_between_two_points(r_hip, r_knee)
    average_leg_len = np.mean([left_leg, right_leg])
    middle_len = length_between_two_points(hip, belly)
    upper_body_len = length_between_two_points(belly, neck)
    head_len = length_between_two_points(head, neck)
    return average_leg_len + middle_len + upper_body_len + head_len


def total_one_side_length(foot, knee, side_hip, hip, hand, elbow, shoulder, neck):
    leg_len = length_between_two_points(knee, foot) + length_between_two_points(side_hip, knee)
    hip_len = length_between_two_points(hip, side_hip)
    arm_len = length_between_two_points(hand, elbow) + length_between_two_points(shoulder, elbow)
    shoulder_len = length_between_two_points(neck, shoulder)
    return leg_len + hip_len + arm_len + shoulder_len


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norm_product
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def armrests(lh, le, ls, n, rs, re, rh):
    l_arm = length_between_two_points(lh, le) + length_between_two_points(ls, le) + length_between_two_points(n, ls)
    r_arm = length_between_two_points(rh, re) + length_between_two_points(rs, re) + length_between_two_points(n, rs)
    return l_arm + r_arm

def score0_3(test_val,min_3,max_3,min_2_low,max_2_low,min_2_high,max_2_high,min_1_low,max_1_low,min_1_high,max_1_high):
    if (min_3<=test_val and test_val<=max_3):
        return 3
    if (min_2_low<=test_val and test_val<=max_2_low):
        return 2.5
    if (min_2_high is not None):
        if (min_2_high <= test_val and test_val <= max_2_high):
            return 2.5
    if (min_1_low<=test_val and test_val<=max_1_low):
        return 1
    if (min_1_high is not None):
        if (min_1_high <= test_val and test_val <= max_1_high):
            return 1
    return 0
def score_per_frame(skeleton_before, skeleton_now):
    if (skeleton_now is None or len(skeleton_now) == 0):
        return 0
    score = 0
    max_score=0
    hip_now, r_hip_now, r_knee_now, r_foot_now, l_hip_now, l_knee_now, l_foot_now, belly_now, neck_now, nose_now, \
        head_now, l_shoulder_now, l_elbow_now, l_hand_now, r_shoulder_now, r_elbow_now, r_hand_now = skeleton_now

    # lengths_now
    left_side_len_now = total_one_side_length(l_foot_now, l_knee_now, l_hip_now, hip_now,
                                              l_hand_now, l_elbow_now, l_shoulder_now, neck_now)
    right_side_len_now = total_one_side_length(r_foot_now, r_knee_now, r_hip_now, hip_now,
                                               r_hand_now, r_elbow_now, r_shoulder_now, neck_now)
    height_now = human_height(hip_now, r_hip_now, r_knee_now, r_foot_now, l_hip_now, l_knee_now, l_foot_now,
                              belly_now, neck_now, head_now)

    shoulder_to_shoulder_now = length_between_two_points(l_shoulder_now, r_shoulder_now)

    l_arm_now = length_between_two_points(l_hand_now, l_elbow_now) + length_between_two_points(l_shoulder_now,
                                                                                               l_elbow_now) + length_between_two_points(
        neck_now, l_shoulder_now)
    r_arm_now = length_between_two_points(r_hand_now, r_elbow_now) + length_between_two_points(r_shoulder_now,
                                                                                               r_elbow_now) + length_between_two_points(
        neck_now, r_shoulder_now)

    l_leg_now = length_between_two_points(l_foot_now, l_knee_now) + length_between_two_points(l_hip_now, l_knee_now)
    r_leg_now = length_between_two_points(r_foot_now, r_knee_now) + length_between_two_points(r_hip_now, r_knee_now)

    back_now = length_between_two_points(hip_now, belly_now) + length_between_two_points(belly_now, neck_now)

    avg_foot_now = torch.mean(torch.stack([l_foot_now, r_foot_now]), dim=0)
    hip2head_now = length_between_two_points(hip_now, head_now)
    hip2foot_now = length_between_two_points(hip_now, avg_foot_now)

    head_len_now = length_between_two_points(head_now, neck_now)

    average_leg_len_now = (l_leg_now + r_leg_now)/2

    average_arm_len_now = (l_arm_now + r_arm_now)/2

    l_down_hand_now = length_between_two_points(l_hand_now, l_elbow_now)
    l_up_hand_now = length_between_two_points(l_shoulder_now, l_elbow_now)
    r_down_hand_now = length_between_two_points(r_hand_now, r_elbow_now)
    r_up_hand_now = length_between_two_points(r_shoulder_now, r_elbow_now)
    avg_down_hand_now = (l_down_hand_now + r_down_hand_now)/2
    avg_up_hand_now = (l_up_hand_now + r_up_hand_now)/2

    l_down_leg_now = length_between_two_points(l_foot_now, l_knee_now)
    l_up_leg_now = length_between_two_points(l_hip_now, l_knee_now)
    r_down_leg_now = length_between_two_points(r_foot_now, r_knee_now)
    r_up_leg_now = length_between_two_points(r_hip_now, r_knee_now)
    avg_down_leg_now = (l_down_leg_now + r_down_leg_now)/2
    avg_up_leg_now = (l_up_leg_now + r_up_leg_now)/2

    head2nose_now = length_between_two_points(head_now, nose_now)
    nose2neck_now = length_between_two_points(nose_now,neck_now)
    l_shoulder2neck_now = length_between_two_points(l_shoulder_now, neck_now)
    r_shoulder2neck_now = length_between_two_points(r_shoulder_now, neck_now)
    neck2belly_now = length_between_two_points(neck_now, belly_now)
    belly2hip_now = length_between_two_points(belly_now, hip_now)
    lhip2hip_now = length_between_two_points(l_hip_now, hip_now)
    rhip2hip_now = length_between_two_points(r_hip_now, hip_now)

    # ratios now
    l_side2r_side_now = left_side_len_now / right_side_len_now
    if l_side2r_side_now > 1: l_side2r_side_now = 1 / l_side2r_side_now
    shoulder_width2height_now = shoulder_to_shoulder_now / height_now
    if shoulder_width2height_now > 1: shoulder_width2height_now = 1 / shoulder_width2height_now
    l_arm2height_now = l_arm_now / height_now
    if l_arm2height_now > 1: l_arm2height_now = 1 / l_arm2height_now
    r_arm2height_now = r_arm_now / height_now
    if r_arm2height_now > 1: r_arm2height_now = 1 / r_arm2height_now
    l_leg2height_now = l_leg_now / height_now
    if l_leg2height_now > 1: l_leg2height_now = 1 / l_leg2height_now
    r_leg2height_now = r_leg_now / height_now
    if r_leg2height_now > 1: r_leg2height_now = 1 / r_leg2height_now
    back2height_now = back_now / height_now
    if back2height_now > 1: back2height_now = 1 / back2height_now
    up2down_now = hip2head_now / hip2foot_now
    if up2down_now > 1: up2down_now = 1 / up2down_now
    head2height_now = head_len_now / height_now
    if head2height_now > 1: head2height_now = 1 / head2height_now
    back2shoulders_now = shoulder_to_shoulder_now / back_now
    if back2shoulders_now > 1: back2shoulders_now = 1 / back2shoulders_now
    avg_leg2back_now = average_leg_len_now / back_now
    if avg_leg2back_now > 1: avg_leg2back_now = 1 / avg_leg2back_now

    avg_arm2back_now = average_arm_len_now / back_now
    if avg_arm2back_now > 1: avg_arm2back_now = 1 / avg_arm2back_now
    upHand2downHand_now = avg_up_hand_now / avg_down_hand_now
    if upHand2downHand_now > 1: upHand2downHand_now = 1 / upHand2downHand_now
    upLeg2downLeg_now = avg_up_leg_now / avg_down_leg_now
    if upLeg2downLeg_now > 1: upLeg2downLeg_now = 1 / upLeg2downLeg_now
    # angles now
    l_elbow_angle_now = calculate_angle(l_hand_now, l_elbow_now, l_shoulder_now)
    r_elbow_angle_now = calculate_angle(r_hand_now, r_elbow_now, r_shoulder_now)
    l_knee_angle_now = calculate_angle(l_foot_now, l_knee_now, l_hip_now)
    r_knee_angle_now = calculate_angle(r_foot_now, r_knee_now, r_hip_now)

    l_head_neck_shoulder_angle_now = calculate_angle(head_now, neck_now, l_shoulder_now)
    r_head_neck_shoulder_angle_now = calculate_angle(head_now, neck_now, r_shoulder_now)
    backbone_angle_now = calculate_angle(neck_now, belly_now, hip_now)
    shoulders_xy_angle_now = angle_with_xy(r_shoulder_now, l_shoulder_now)

    r_shoulder2elbow_z_angle_now = angle_with_z(r_elbow_now, r_shoulder_now)
    l_shoulder2elbow_z_angle_now = angle_with_z(l_elbow_now, l_shoulder_now)
    neck_hip_z_angle_now = angle_with_xy(neck_now, hip_now)
    l_hip_down_angle_now = calculate_angle(l_knee_now, l_hip_now, hip_now)
    r_hip_down__angle_now = calculate_angle(r_knee_now, r_hip_now, hip_now)
    l_hip_up_angle_now = calculate_angle(l_hip_now, hip_now, belly_now)
    r_hip_up_angle_now = calculate_angle(r_hip_now, hip_now, belly_now)
    belly_angle_now = calculate_angle(hip_now, belly_now, neck_now)
    l_shoulder_angle_now = calculate_angle(neck_now, l_shoulder_now, l_elbow_now)
    l_neck_angle_now = calculate_angle(belly_now, neck_now, l_shoulder_now)
    r_shoulder_angle_now = calculate_angle(neck_now, r_shoulder_now, r_elbow_now)
    r_neck_angle_now = calculate_angle(belly_now, neck_now, r_shoulder_now)

    score+=score0_3(l_side2r_side_now, 0.96, 1, .9, .96, None, None, .5, .9, None, None)
    max_score+=3
    score+=score0_3(shoulder_width2height_now, 0.32, .4, .29, .32, None, None, .25, .29, None, None)
    max_score+=3
    score += score0_3(l_arm2height_now, 0.52, .56, .51, .52, .57, .58, .48, .49, .58, .59)
    max_score += 3
    score += score0_3(r_arm2height_now, 0.53, .59, .52, .53, .59, .6, .51, .52, .6, .61)
    max_score += 3
    score += score0_3(l_leg2height_now,  0.53, .59, .52, .53, .59, .6, .51, .52, .6, .61)
    max_score += 3
    score += score0_3(r_leg2height_now, 0.54, .56, .52, .54, .56, .58, .5, .52, .58, .6)
    max_score+=3
    score += score0_3(back2height_now, 0.28, .3, .25, .28, None, None, .2, .25, None, None)
    max_score += 3
    score += score0_3(up2down_now, 0.8, .9, .7, .8, None, None, .6, .7, None, None)
    max_score += 3
    score += score0_3(head2height_now, 0.125, .15, .11, .125, .15, .16, .1, .11, .17, .17)
    max_score += 3
    score+=score0_3(back2shoulders_now, 0.95, 1, .9, .95, None, None, .85, .9, None, None)
    max_score+=3
    score += score0_3(avg_leg2back_now, 0.52, .58, .505, .52, .58, .595, .49, .505, .594, .61)
    max_score += 3
    score += score0_3(avg_arm2back_now, 0.53, .56, .5, .53, .56, .59, .48, .5, .59, .61)
    max_score += 3
    score += score0_3(upHand2downHand_now, 0.98, 1, .95, .98, None, None, .92, .95, None, None)
    max_score += 3
    score+=score0_3(upLeg2downLeg_now, 0.85, .9, .8, .85, .9, .95, .75, .8, None, None)
    max_score+=3
    score += score0_3(l_elbow_angle_now, 130, 170, 110, 130, 170, 175, 80, 110, 175, 180)
    max_score += 3
    score += score0_3(r_elbow_angle_now, 130, 160, 110, 130, 160, 175, 80, 110, 175, 180)
    max_score += 3
    score += score0_3(l_knee_angle_now, 140, 175, 130, 140, 175, 178, 120, 130, 178, 180)
    max_score += 3
    score+=score0_3(r_knee_angle_now, 140, 175, 130, 140, 175, 178, 120, 130, 178, 180)
    max_score+=3
    score += score0_3(l_head_neck_shoulder_angle_now, 85, 95, 80, 85, 95, 110, 75, 80, 110, 115)
    max_score += 3
    score += score0_3(r_head_neck_shoulder_angle_now, 75, 90, 71, 75, 90, 94, 67, 71, 94, 98)
    max_score += 3
    score += score0_3(backbone_angle_now, 163, 170, 155, 163, 170, 178, 145, 155, 178, 181)
    max_score += 3
    score+=score0_3(shoulders_xy_angle_now, -6, 6, -10, -6, 6, 10, -15, -10, 10, 15)
    max_score+=3
    score += score0_3(r_shoulder2elbow_z_angle_now, 33, 37, 28, 33, 37, 42, 25, 28, 42, 45)
    max_score += 3
    score+=score0_3(l_shoulder2elbow_z_angle_now, 33, 37, 28, 33, 37, 42, 25, 28, 42, 45)
    max_score+=3
    score += score0_3(neck_hip_z_angle_now, -77, -72, -76, -70, -72, -69, -79, -76, -69, -66)
    max_score += 3



    if (skeleton_before is not None) and (len(skeleton_before) == 17):
        hip_before, r_hip_before, r_knee_before, r_foot_before, l_hip_before, l_knee_before, l_foot_before, belly_before, \
            neck_before, nose_before, head_before, l_shoulder_before, l_elbow_before, l_hand_before, r_shoulder_before, \
            r_elbow_before, r_hand_before = skeleton_before

        # lengths_before

        l_down_hand_before = length_between_two_points(l_hand_before, l_elbow_before)
        l_up_hand_before = length_between_two_points(l_shoulder_before, l_elbow_before)
        r_down_hand_before = length_between_two_points(r_hand_before, r_elbow_before)
        r_up_hand_before = length_between_two_points(r_shoulder_before, r_elbow_before)
        l_down_leg_before = length_between_two_points(l_foot_before, l_knee_before)
        l_up_leg_before = length_between_two_points(l_hip_before, l_knee_before)
        r_down_leg_before = length_between_two_points(r_foot_before, r_knee_before)
        r_up_leg_before = length_between_two_points(r_hip_before, r_knee_before)
        head2nose_before = length_between_two_points(head_before, nose_before)
        nose2neck_before = length_between_two_points(nose_before, neck_before)
        l_shoulder2neck_before = length_between_two_points(l_shoulder_before, neck_before)
        r_shoulder2neck_before = length_between_two_points(r_shoulder_before, neck_before)
        neck2belly_before = length_between_two_points(neck_before, belly_before)
        belly2hip_before = length_between_two_points(belly_before, hip_before)
        lhip2hip_before = length_between_two_points(l_hip_before, hip_before)
        rhip2hip_before = length_between_two_points(r_hip_before, hip_before)

        # ratios before
        neck2belly = neck2belly_now / neck2belly_before
        l_down_hand = (l_down_hand_now/l_down_hand_before) / neck2belly
        if l_down_hand>1: down_hand=1/l_down_hand
        l_up_hand = l_up_hand_now / l_up_hand_before / neck2belly
        if l_up_hand>1: l_up_hand=1/l_up_hand
        r_down_hand = r_down_hand_now / r_down_hand_before / neck2belly
        if r_down_hand>1: r_down_hand=1/r_down_hand
        r_up_hand = r_up_hand_now / r_up_hand_before / neck2belly
        if r_up_hand>1: r_up_hand=1/r_up_hand
        l_down_leg = l_down_leg_now / l_down_leg_before / neck2belly
        if l_down_leg>1: l_down_leg=1/l_down_leg
        l_up_leg = l_up_leg_now / l_up_leg_before / neck2belly
        if l_up_leg>1: l_up_leg=1/l_up_leg
        r_down_leg = r_down_leg_now / r_down_leg_before / neck2belly
        if r_down_leg>1: r_down_leg=1/r_down_leg
        r_up_leg = r_up_leg_now / r_up_leg_before / neck2belly
        if r_up_leg>1: r_up_leg=1/r_up_leg
        head2nose = head2nose_now / head2nose_before / neck2belly
        if head2nose>1: head2nose=1/head2nose
        nose2neck = nose2neck_now / nose2neck_before / neck2belly
        if nose2neck>1: nose2neck=1/nose2neck
        l_shoulder2neck = l_shoulder2neck_now / l_shoulder2neck_before / neck2belly
        if l_shoulder2neck>1: l_shoulder2neck=1/l_shoulder2neck
        r_shoulder2neck = r_shoulder2neck_now / r_shoulder2neck_before / neck2belly
        if r_shoulder2neck>1: r_shoulder2neck=1/r_shoulder2neck
        belly2hip = belly2hip_now / belly2hip_before / neck2belly
        if belly2hip>1: belly2hip=1/belly2hip
        lhip2hip = lhip2hip_now / lhip2hip_before / neck2belly
        if lhip2hip>1: lhip2hip=1/lhip2hip
        rhip2hip = rhip2hip_now / rhip2hip_before / neck2belly
        if rhip2hip>1: rhip2hip=1/rhip2hip

        # angles_before
        l_elbow_angle_before = calculate_angle(l_hand_before, l_elbow_before, l_shoulder_before)
        r_elbow_angle_before = calculate_angle(r_hand_before, r_elbow_before, r_shoulder_before)
        l_knee_angle_before = calculate_angle(l_foot_before, l_knee_before, l_hip_before)
        r_knee_angle_before = calculate_angle(r_foot_before, r_knee_before, r_hip_before)
        l_hip_down_angle_before = calculate_angle(l_knee_before, l_hip_before, hip_before)
        r_hip_down__angle_before = calculate_angle(r_knee_before, r_hip_before, hip_before)
        l_hip_up_angle_before = calculate_angle(l_hip_before, hip_before, belly_before)
        r_hip_up_angle_before = calculate_angle(r_hip_before, hip_before, belly_before)
        belly_angle_before = calculate_angle(hip_before, belly_before, neck_before)
        l_shoulder_angle_before = calculate_angle(neck_before, l_shoulder_before, l_elbow_before)
        l_neck_angle_before = calculate_angle(belly_before, neck_before, l_shoulder_before)
        r_shoulder_angle_before = calculate_angle(neck_before, r_shoulder_before, r_elbow_before)
        r_neck_angle_before = calculate_angle(belly_before, neck_before, r_shoulder_before)

        l_elbow_angle_difference = np.abs(l_elbow_angle_before-l_elbow_angle_now)
        r_elbow_angle_difference = np.abs(r_elbow_angle_before-r_elbow_angle_now)
        l_knee_angle_difference = np.abs(l_knee_angle_before-l_knee_angle_now)
        r_knee_angle_difference = np.abs(r_knee_angle_before-r_knee_angle_now)
        l_hip_down_angle_difference = np.abs(l_hip_down_angle_before-l_hip_down_angle_now)
        r_hip_down__angle_difference = np.abs(r_hip_down__angle_before-r_hip_down__angle_now)
        l_hip_up_angle_difference = np.abs(l_hip_up_angle_before-l_hip_up_angle_now)
        r_hip_up_angle_difference = np.abs(r_hip_up_angle_before-r_hip_up_angle_now)
        belly_angle_difference = np.abs(belly_angle_before-belly_angle_now)
        l_shoulder_angle_difference = np.abs(l_shoulder_angle_before-l_shoulder_angle_now)
        l_neck_angle_difference = np.abs(l_neck_angle_before-l_neck_angle_now)
        r_shoulder_angle_difference = np.abs(r_shoulder_angle_before-r_shoulder_angle_now)
        r_neck_angle_difference = np.abs(r_neck_angle_before-r_neck_angle_now)

        f_jump = 2
        f_jump = 2
        score += f_jump * score0_3(l_down_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_up_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_down_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_up_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_down_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_up_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_down_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_up_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(head2nose, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(nose2neck, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_shoulder2neck, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_shoulder2neck, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(belly2hip, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(lhip2hip, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(rhip2hip, .97, 1, .92, .97, None, None, .85, .92, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_elbow_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_elbow_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_knee_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_knee_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_hip_down_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_hip_down__angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_hip_up_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_hip_up_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(belly_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_shoulder_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(l_neck_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_shoulder_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3
        score += f_jump * score0_3(r_neck_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        max_score += f_jump * 3

        angle_decrease_0 = 0
        angle_decrease_l_0 = 0
        angle_decrease_r_0 = 0
        ratio_decrease_0 = 0
        ratio_decrease_l_0 = 0
        ratio_decrease_r_0 = 0
        angle_decrease_1 = 0
        angle_decrease_l_1 = 0
        angle_decrease_r_1 = 0
        ratio_decrease_1 = 0
        ratio_decrease_l_1 = 0
        ratio_decrease_r_1 = 0

        angle_decrease_1_counter = 0
        angle_decrease_l_1_counter = 0
        angle_decrease_r_1_counter = 0
        ratio_decrease_1_counter = 0
        ratio_decrease_l_1_counter = 0
        ratio_decrease_r_1_counter = 0

        l_down_hand_score = score0_3(l_down_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if l_down_hand_score:
            ratio_decrease_1+=1
            ratio_decrease_l_1=1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_l_0 = 1

        l_up_hand_score = score0_3(l_up_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if l_up_hand_score:
            ratio_decrease_1 += 1
            ratio_decrease_l_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_l_0 = 1
        r_down_hand_score = score0_3(r_down_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if r_down_hand_score:
            ratio_decrease_1 += 1
            ratio_decrease_r_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_r_0 = 1
        r_up_hand_score = score0_3(r_up_hand, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if r_up_hand_score:
            ratio_decrease_1 += 1
            ratio_decrease_r_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_r_0 = 1
        l_down_leg_score = score0_3(l_down_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if l_down_leg_score:
            ratio_decrease_1 += 1
            ratio_decrease_l_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_l_0 = 1
        l_up_leg_score = score0_3(l_up_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if l_up_leg_score:
            ratio_decrease_1 += 1
            ratio_decrease_l_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_l_0 = 1
        r_down_leg_score = score0_3(r_down_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if r_down_leg_score:
            ratio_decrease_1 += 1
            ratio_decrease_r_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_r_0 = 1
        r_up_leg_score = score0_3(r_up_leg, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if r_up_leg_score:
            ratio_decrease_1 += 1
            ratio_decrease_r_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_r_0 = 1
        head2nose_score = score0_3(head2nose, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if head2nose_score:
            ratio_decrease_1+=1
        else:
            ratio_decrease_0+=1
        nose2neck_score = score0_3(nose2neck, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if nose2neck_score:
            ratio_decrease_1+=1
        else:
            ratio_decrease_0+=1
        l_shoulder2neck_score = score0_3(l_shoulder2neck, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if l_shoulder2neck_score:
            ratio_decrease_1 += 1
            ratio_decrease_l_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_l_0 = 1
        r_shoulder2neck_score = score0_3(r_shoulder2neck, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if r_shoulder2neck_score:
            ratio_decrease_1 += 1
            ratio_decrease_r_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_r_0 = 1
        belly2hip_score = score0_3(belly2hip, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if belly2hip_score:
            ratio_decrease_1+=1
        else:
            ratio_decrease_0+=1
        lhip2hip_score = score0_3(lhip2hip, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if lhip2hip_score:
            ratio_decrease_1 += 1
            ratio_decrease_l_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_l_0 = 1
        rhip2hip_score = score0_3(rhip2hip, .97, 1, .92, .97, None, None, .85, .92, None, None)
        if rhip2hip_score:
            ratio_decrease_1 += 1
            ratio_decrease_r_1 = 1
        else:
            ratio_decrease_0 += 1
            ratio_decrease_r_0 = 1
        l_elbow_angle_difference_score = score0_3(l_elbow_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if l_elbow_angle_difference_score:
            angle_decrease_1+=1
            angle_decrease_l_1+=1
        else:
            angle_decrease_0+=1
            angle_decrease_l_0+=1
        r_elbow_angle_difference_score = score0_3(r_elbow_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if r_elbow_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_r_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_r_0 += 1
        l_knee_angle_difference_score = score0_3(l_knee_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if l_knee_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_l_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_l_0 += 1
        r_knee_angle_difference_score = score0_3(r_knee_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if r_knee_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_r_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_r_0 += 1
        l_hip_down_angle_difference_score = score0_3(l_hip_down_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if l_hip_down_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_l_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_l_0 += 1
        r_hip_down__angle_difference_score = score0_3(r_hip_down__angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if r_hip_down__angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_r_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_r_0 += 1
        l_hip_up_angle_difference_score = score0_3(l_hip_up_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if l_hip_up_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_l_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_l_0 += 1
        r_hip_up_angle_difference_score = score0_3(r_hip_up_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if r_hip_up_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_r_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_r_0 += 1
        belly_angle_difference_score = score0_3(belly_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if belly_angle_difference_score:
            angle_decrease_1+=1
        else:
            angle_decrease_0+=1
        l_shoulder_angle_difference_score = score0_3(l_shoulder_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if l_shoulder_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_l_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_l_0 += 1
        l_neck_angle_difference_score = score0_3(l_neck_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if l_neck_angle_difference_score:
            angle_decrease_1+=1
        else:
            angle_decrease_0+=1
        r_shoulder_angle_difference_score = score0_3(r_shoulder_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if r_shoulder_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_r_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_r_0 += 1
        r_neck_angle_difference_score = score0_3(r_neck_angle_difference, 0, 3.5, 3.5, 5, None, None, 5, 9, None, None)
        if r_neck_angle_difference_score:
            angle_decrease_1 += 1
            angle_decrease_r_1 += 1
        else:
            angle_decrease_0 += 1
            angle_decrease_r_0 += 1

        score += angle_decrease_1 * 0.2
        score -= angle_decrease_0 * 3
        score -= 4 *((((angle_decrease_l_1*0.25) + angle_decrease_l_0) + ((angle_decrease_r_1 * 0.25) +
                                                                          angle_decrease_r_0)))/2
        score += ratio_decrease_1 * 0.4
        score -= ratio_decrease_0 * 3.5
        score -= 5 * ((((ratio_decrease_l_1 * 0.5) + ratio_decrease_l_0) + ((ratio_decrease_r_1* 0.5) +
                                                                            ratio_decrease_r_0)))/2
        return score/max_score

def video_main(tensors_array):
    test = 0
    for i in range(len(tensors_array)):
        if i==0:
            previous_tensor=None
        else:
            previous_tensor = tensors_array[i-1]
        spf =score_per_frame(previous_tensor, tensors_array[i])
        if (spf is not None):
            test += score_per_frame(previous_tensor, tensors_array[i])
        else:
            test += 0
    average_test = test/len(tensors_array)
    print(f"test = {average_test}")
    if type(average_test)== torch.Tensor:
        average_test=average_test.item()
    return average_test


i = 0
tests = []
for vid_name, vid_array in vid_dic.items():
    print(f"\n{vid_name}:")
    tests.append(video_main(vid_array))
    i += 1
write_to_excel(tests)