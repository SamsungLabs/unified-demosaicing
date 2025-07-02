"""
Author(s):
Abdelrahman Abdelhamed

Camera pipeline utilities.
"""

import os
from fractions import Fraction

import scipy
import cv2
import numpy as np
import exifread
import math
# from exifread import Ratio
from exifread.utils import Ratio
import rawpy
from scipy.io import loadmat
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import struct

from .dng_opcode import parse_opcode_lists
from .exif_data_formats import exif_formats
from .exif_utils import parse_exif_tag, parse_exif, get_tag_values_from_ifds


#raw_image.raw_image[54:4544,148:6868]
#[54, 148, 4544, 6868]

#raw image shape is 4544x6880

def get_visible_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image_visible
    #x = raw_image.raw_image[54:4544,148:6719]
    # raw_image = rawpy.imread(image_path).raw_image.copy()
    return raw_image


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    metadata['linearization_table'] = get_linearization_table(tags, ifds)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags, ifds)
    color_matrix_1, color_matrix_2 = get_color_matrices(tags, ifds)
    metadata['camera_calibration_1'], metadata['camera_calibration_2'] = get_calibration_matrices(tags, ifds)
    metadata['analog_balance'] = get_analog_balance(tags, ifds)
    forward_matrix_1, forward_matrix_2 = get_forward_matrices(tags, ifds)
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    metadata['forward_matrix_1'] = forward_matrix_1
    metadata['forward_matrix_2'] = forward_matrix_2
    metadata['orientation'] = get_orientation(tags, ifds)
    metadata['noise_profile'] = get_noise_profile(tags, ifds)
    metadata['hsv_lut'] = get_hsv_luts(tags, ifds)
    metadata['profile_lut'] = get_profile_luts(tags, ifds)
    # ...

    # opcode lists
    metadata['opcode_lists'] = parse_opcode_lists(ifds)

#    metadata['polynomial'] = metadata['opcode_lists'][51009]


    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata['as_shot_neutral'] is None:
        metadata['as_shot_neutral'] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        metadata['color_matrix_1'] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['color_matrix_2'] is None:
        metadata['color_matrix_2'] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_linearization_table(tags, ifds):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712', 'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714', 'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Black level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50714, ifds)
    return vals


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717', 'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags, ifds):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728', 'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)


def get_hsv_luts(tags, ifds):
    hsv_lut = None
    possible_keys_1 = ['Image Tag 0xC6F9', 'Image Tag 50937', 'ProfileHueSatMapDims', 'Image ProfileHueSatMapDims']
    hue_sat_map_dims = get_values(tags, possible_keys_1)
    if hue_sat_map_dims is None:
        hue_sat_map_dims = get_tag_values_from_ifds(50937, ifds)
    possible_keys_2 = ['Image Tag 0xC6FA', 'Image Tag 50938', 'ProfileHueSatMapData1', 'Image ProfileHueSatMapData1']
    hsv_lut_1 = None
    if hsv_lut_1 is None:
        hsv_lut_1 = get_tag_values_from_ifds(50939, ifds)

    if(hue_sat_map_dims is not None and hsv_lut_1 is not None):
        hue_sat_map_dims.append(3)
        hsv_lut = np.reshape(hsv_lut_1, newshape = hue_sat_map_dims)

    return hsv_lut

def get_profile_luts(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC725', 'Image Tag 50981', 'ProfileLookTableDims', 'Image ProfileLookTableDims']
    profile_dims = get_values(tags, possible_keys_1)
    if profile_dims is None:
        profile_dims = get_tag_values_from_ifds(50981, ifds)
    possible_keys_2 = ['Image Tag 0xC726', 'Image Tag 50982', 'ProfileLookTableData', 'Image ProfileLookTableData']
    profile_lut = None
    if profile_lut is None:
        profile_lut = get_tag_values_from_ifds(50982, ifds)

    if (profile_dims is not None and profile_lut is not None):
        profile_dims.append(3)
        profile_lut = np.reshape(profile_lut, newshape=profile_dims)


    return profile_lut

def get_color_matrices(tags, ifds):

    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721', 'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722', 'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    return color_matrix_1, color_matrix_2

def get_forward_matrices(tags, ifds):

    possible_keys_1 = ['Image Tag 0xC714', 'Image Tag 50964', 'ForwardMatrix1', 'Image ForwardMatrix1']
    forward_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC715', 'Image Tag 50965', 'ForwardMatrix2', 'Image ForwardMatrix2']
    forward_matrix_2 = get_values(tags, possible_keys_2)
    return forward_matrix_1, forward_matrix_2

def get_calibration_matrices(tags, ifds):

    possible_keys_1 = ['Image Tag 0xC623', 'Image Tag 50723', 'CameraCalibration1', 'Image CameraCalibration1']
    camera_calibration_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC624', 'Image Tag 50724', 'CameraCalibration2', 'Image CameraCalibration2']
    camera_calibration_2 = get_values(tags, possible_keys_2)
    return camera_calibration_1, camera_calibration_2

def get_analog_balance(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC627', 'Image Tag 50727', 'AnalogBalance', 'Image AnalogBalance']
    analog_balance = get_values(tags, possible_keys_1)
    return analog_balance

def get_orientation(tags, ifds):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041', 'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals



def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def normalize(raw_image, black_level, white_level):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

    # special case for pixelshift cameras
    if type(white_level) is list and len(white_level) == 3 and raw_image.shape[2] == 4:
        white_level_m = np.zeros(raw_image.shape)
        white_level_m[:, :, 0] = float(white_level[0])
        white_level_m[:, :, 1] = float(white_level[1])
        white_level_m[:, :, 2] = float(white_level[2])
        white_level = white_level_m

    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

    if type(black_level) is list and len(black_level) == 12:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
                   [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1],
                   [0, 0, 2], [0, 1, 2], [1, 0, 2], [1, 1, 2]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

    normalized_image = raw_image.astype(np.float32) - black_level_mask

    # if some values were smaller than black level
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats

def polynomial(current_image, polynomial_opcode):
    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    po = polynomial_opcode[8]
    for i, idx in enumerate(idx2by2):
        if(i == 0):
            po_i_data = po[0].data
        elif(i==1 or i==2):
            po_i_data = po[1].data
        else:
            po_i_data = po[2].data
        plane = current_image[idx[0]::2, idx[1]::2]
        coefficient = po_i_data["Coefficient"]
        planeTransform = coefficient[3]* np.power(plane,3) + coefficient[2]* np.power(plane,2)+(coefficient[1])* np.power(plane,1)+coefficient[0]
        current_image[idx[0]::2, idx[1]::2] = planeTransform
    return current_image

def vignetting_correction(raw_image, vignetting_opcode):
    data = vignetting_opcode.data
    k0 = struct.unpack('>d', data[0:8])[0]
    k1 = struct.unpack('>d', data[8:16])[0]
    k2 = struct.unpack('>d', data[16:24])[0]
    k3 = struct.unpack('>d', data[24:32])[0]
    k4 = struct.unpack('>d', data[32:40])[0]
    cx_hat = struct.unpack('>d', data[40:48])[0]
    cy_hat = struct.unpack('>d', data[48:56])[0]

    #pixel coordinates of top left pixel
    x0 = 0
    y0 = 0

    #pixel coordinates of bottom right pixel
    x1 = raw_image.shape[0]
    y1 = raw_image.shape[1]

    cx = x0 + cx_hat*(x1-x0)
    cy = y0 + cy_hat*(y1-y0)

    mx = max(abs(x0-cx), abs(x1-cx))
    my = max(abs(y0-cy), abs(y1-cy))

    m = math.sqrt(pow(mx, 2) + pow(my, 2))

    meshgrid = np.mgrid[0:raw_image.shape[0],0:raw_image.shape[1]]

    x = meshgrid[0]
    y = meshgrid[1]

    sum = np.square(x - cx) + np.square(y - cy)
    r = (1/m)*np.sqrt(sum)
    g = k0 * np.power(r, 2) + k1*np.power(r,4) + k2*np.power(r,6) + k3*np.power(r,8) + k4*np.power(r,10)
    g = g+1
    for c in range(3):
        raw_image[:,:,c] = raw_image[:,:,c]* g
    return raw_image



def lens_shading_correction(raw_image, gain_map_opcode, bayer_pattern, gain_map=None, clip=True):
    """
    Apply lens shading correction map.
    :param raw_image: Input normalized (in [0, 1]) raw image.
    :param gain_map_opcode: Gain map opcode.
    :param bayer_pattern: Bayer pattern (RGGB, GRBG, ...).
    :param gain_map: Optional gain map to replace gain_map_opcode. 1 or 4 channels in order: R, Gr, Gb, and B.
    :param clip: Whether to clip result image to [0, 1].
    :return: Image with gain map applied; lens shading corrected.
    """

    if gain_map is None and gain_map_opcode:
        gain_map = gain_map_opcode.data['map_gain_2d']

    # resize gain map, make it 4 channels, if needed
    gain_map = cv2.resize(gain_map, dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
                          interpolation=cv2.INTER_LINEAR)
    if len(gain_map.shape) == 2:
        gain_map = np.tile(gain_map[..., np.newaxis], [1, 1, 4])

    if gain_map_opcode:
        # TODO: consider other parameters

        top = gain_map_opcode.data['top']
        left = gain_map_opcode.data['left']
        bottom = gain_map_opcode.data['bottom']
        right = gain_map_opcode.data['right']
        rp = gain_map_opcode.data['row_pitch']
        cp = gain_map_opcode.data['col_pitch']

        gm_w = right - left
        gm_h = bottom - top

        # gain_map = cv2.resize(gain_map, dsize=(gm_w, gm_h), interpolation=cv2.INTER_LINEAR)

        # TODO
        # if top > 0:
        #     pass
        # elif left > 0:
        #     left_col = gain_map[:, 0:1]
        #     rep_left_col = np.tile(left_col, [1, left])
        #     gain_map = np.concatenate([rep_left_col, gain_map], axis=1)
        # elif bottom < raw_image.shape[0]:
        #     pass
        # elif right < raw_image.shape[1]:
        #     pass

    result_image = raw_image.copy()

    # one channel
    # result_image[::rp, ::cp] *= gain_map[::rp, ::cp]

    # per bayer channel
    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    bayer_pattern_idx = np.array(bayer_pattern)
    # blue channel index --> 3
    bayer_pattern_idx[bayer_pattern_idx == 2] = 3
    # second green channel index --> 2
    if bayer_pattern_idx[3] == 1:
        bayer_pattern_idx[3] = 2
    else:
        bayer_pattern_idx[2] = 2
    for c in range(4):
        i0 = upper_left_idx[c][0]
        j0 = upper_left_idx[c][1]
        result_image[i0::2, j0::2] *= gain_map[:, :, bayer_pattern_idx[c]]

    if clip:
        result_image = np.clip(result_image, 0.0, 1.0)

    return result_image


def white_balance(normalized_image, as_shot_neutral, cfa_pattern):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)
    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    white_balanced_image = np.zeros(normalized_image.shape)
    for i, idx in enumerate(idx2by2):
        idx_y = idx[0]
        idx_x = idx[1]
        white_balanced_image[idx_y::step2, idx_x::step2] = \
            normalized_image[idx_y::step2, idx_x::step2] / as_shot_neutral[cfa_pattern[i]]
    #no clipping
    #white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image
    #return normalized_image


def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    # using opencv edge-aware demosaicing
    if alg_type != '':
        alg_type = '_' + alg_type
    if output_channel_order == 'BGR':
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
            print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
            print("CFA pattern not identified.")
    return opencv_demosaic_flag


def performInterpolation(xyz_image, lut_table):
    def interp(image_in, table):
        #using the actual image that comes in creates some problems
        image_copy = np.copy(image_in)

        image_out = np.copy(image_copy)


        h_div, s_div, v_div, _= table.shape

        if (h_div == 1):
            image_copy[:, :, 0] = image_copy[:, :, 0] * 0

        if (s_div == 1):
            image_copy[:, :, 1] = image_copy[:, :, 1] * 0
        #this is common for there not to be any value divisions
        #for this just make all the values 0 so the interpolation works
        if(v_div == 1):
            image_copy[:, :, 2] = image_copy[:, :, 2] * 0

        #clip these in case they fall out of the range of the table
        image_copy[:, :, 1] = image_copy[:, :, 1].clip(0, 1)
        image_copy[:, :, 2] = image_copy[:, :, 2].clip(0, 1)

        #table = table.reshape(90,30,3)
        #i do wrap around trick for hue
        table_expanded_hue = np.empty((h_div+1,s_div,v_div,3))
        #expand the table to allow "wrap around calculation"
        table_expanded_hue[-1, :, :] = table[0, :, :]
        table_expanded_hue[0:h_div, :, :] = table
        table_expanded = table_expanded_hue

        #as for value we just add an extra dimension to make interp function happy
        table_expanded_val = np.empty((h_div + 1, s_div, v_div+1, 3))
        if(v_div == 1):
           table_expanded_val[:, :, -1] = table_expanded_hue[:, :, 0]
           table_expanded_val[:, :, 0:v_div] = table_expanded_hue
           v_div += 1
           table_expanded = table_expanded_val



        hue_p = np.linspace(0,360,h_div+1)
        sat_p = np.linspace(0, 1, s_div)
        val_p = np.linspace(0, 1, v_div)
        p = (hue_p, sat_p, val_p)
        outInterpolate = scipy.interpolate.interpn(points = p, values = table_expanded, xi = image_copy)
        image_out[:,:,0] =(image_out[:,:,0] + outInterpolate[:,:,0] + 360) % 360 #hue
        image_out[:,:,1] = image_out[:,:,1] * outInterpolate[:,:,1] #sat
        image_out[:,:,2] = image_out[:,:,2] * outInterpolate[:,:,2] #value
        return image_out

    #prophoto
    rgb2xyz = np.array([[0.7976749, 0.1351917, 0.0313534],
                         [0.2880402, 0.7118741, 0.0000857],
                         [0.0000000, 0.0000000, 0.8252100]])

    xyz2rgb = np.linalg.inv(rgb2xyz)

    #CLIPPING  - not sure about this
    xyz_image = np.clip(xyz_image, 0, 1)
    rgb_image = xyz2rgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    rgb_image = np.sum(rgb_image, axis=-1)

    #fix hsv calculation
    rgb = rgb_image.astype('float32')

    #taken from https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python
    def rgb2hsv(rgb):
        """ convert RGB to HSV color space

        :param rgb: np.ndarray
        :return: np.ndarray
        """

        rgb = rgb.astype('float')
        maxv = np.amax(rgb, axis=2)
        maxc = np.argmax(rgb, axis=2)
        minv = np.amin(rgb, axis=2)
        minc = np.argmin(rgb, axis=2)

        hsv = np.zeros(rgb.shape, dtype='float')
        hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
        hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
        hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
        hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
        hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
        hsv[maxv != 0, 1] = ((maxv - minv) / (maxv + np.spacing(1)))[maxv != 0]
        hsv[..., 2] = maxv

        return hsv

    def hsv2rgb(hsv):
        """ convert HSV to RGB color space

        :param hsv: np.ndarray
        :return: np.ndarray
        """

        hi = np.floor(hsv[..., 0] / 60.0) % 6
        hi = hi.astype('uint8')
        v = hsv[..., 2].astype('float')
        f = (hsv[..., 0] / 60.0) - np.floor(hsv[..., 0] / 60.0)
        p = v * (1.0 - hsv[..., 1])
        q = v * (1.0 - (f * hsv[..., 1]))
        t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

        rgb = np.zeros(hsv.shape)
        rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
        rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
        rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
        rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
        rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
        rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]

        return rgb


    hsv = rgb2hsv(rgb)

    #hsvChanged = hsv
    hsvChanged = interp(hsv, lut_table)
    #hsvChanged = lut.apply(hsv, interpolator=interp)

    #hsvChangedInt8 = hsvChanged.astype('uint8')
    rgbChanged = (hsv2rgb(hsvChanged.astype('float32')))
    #rgbChanged = rgbChanged /255.0



    xyz_out = rgb2xyz[np.newaxis, np.newaxis, :, :] * rgbChanged[:, :, np.newaxis, :]
    xyz_out = np.sum(xyz_out, axis=-1)


    #rgb = np.matmul()


    #taken from https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/



    #hsv = cv2.cvtColor(xyz,cv2.COLOR_XYZ2RGB)
    #hsv = cv2.cvtColor(xyz, cv2.COLOR_XYZ2RGB)

    return xyz_out


def performLookTable(xyz_image, lut):
    def interp(samples, lut_table):
        #p = np.mgrid[0:360:4, 0:100:100/30.0]
        #x = p.transpose(1,2,3, 0)
        #p = p.reshape(90,30,1,3)
        #samples = samples.reshape(4490*6720,3)
        #samples = samples.reshape(4490,6720,3)

        samples[:,:,0] = samples[:,:,0]*360
        samples[:,:,1] = samples[:,:,1]
        samples[:,:,2] = samples[:,:,2]
        samplesOriginal = np.copy(samples)
        #samples = samples[:,:,0:2]
        lut_table = lut_table.reshape(36,8,16,3)
        p = (np.linspace(0,360,36), np.linspace(0,1,8), np.linspace(0,1,16))
        outInterpolate = scipy.interpolate.interpn(points = p, values = lut_table, xi = samples)
        samplesOriginal[:,:,0] =(samplesOriginal[:,:,0] )#+ outInterpolate[:,:,0])
        samplesOriginal[:,:,1] = (np.clip(samplesOriginal[:,:,1] * outInterpolate[:,:,1], 0, 1))
        samplesOriginal[:,:,2] = (np.clip(samplesOriginal[:,:,2] * outInterpolate[:,:,2], 0, 1))
        return samplesOriginal

    #prophoto
    rgb2xyz = np.array([[0.7347, 	0.2653, 0.0],
                         [0.1596, 0.8404, 0.0],
                         [0.0366, 0.0001, 0.9633]])

    xyz2rgb = np.linalg.inv(rgb2xyz)

    rgb_image = xyz2rgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    rgb_image = np.sum(rgb_image, axis=-1)


    rgb = rgb_image.astype('float32')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsvChanged = lut.apply(hsv, interpolator=interp)

    #hsvChangedInt8 = hsvChanged.astype('uint8')
    rgbChanged = (cv2.cvtColor(hsvChanged.astype('float32'), cv2.COLOR_HSV2RGB))
    #rgbChanged = rgbChanged /255.0



    xyz_out = rgb2xyz[np.newaxis, np.newaxis, :, :] * rgbChanged[:, :, np.newaxis, :]
    xyz_out = np.sum(xyz_out, axis=-1)


    #rgb = np.matmul()


    #taken from https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/



    #hsv = cv2.cvtColor(xyz,cv2.COLOR_XYZ2RGB)
    #hsv = cv2.cvtColor(xyz, cv2.COLOR_XYZ2RGB)

    return xyz_out

def demosaic(white_balanced_image, cfa_pattern, output_channel_order='RGB', alg_type='VNG'):
    """
    Demosaic a Bayer image.
    :param white_balanced_image:
    :param cfa_pattern:
    :param output_channel_order:
    :param alg_type: 
    qalgorithm type. options: '', 'EA' for edge-aware, 'VNG' for variable number of gradients
    :return: Demosaiced image
    """
    if alg_type == 'VNG':
        max_val = 255
        wb_image = (white_balanced_image * max_val).astype(dtype=np.uint8)
    else:
        max_val = 16383/np.max(white_balanced_image)
        wb_image = (white_balanced_image * max_val).astype(dtype=np.uint16)

    if alg_type in ['', 'EA', 'VNG']:
        opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
        demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
    elif alg_type == 'menon2007':
        cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
        demosaiced_image = demosaicing_CFA_Bayer_Menon2007(wb_image, pattern=cfa_pattern_str)

    demosaiced_image = demosaiced_image.astype(dtype=np.float32)/ max_val

    return demosaiced_image


def apply_color_space_transform(demosaiced_image, color_correction_1, color_correction_2, forward_matrix_1, forward_matrix_2, as_shot_neutral, analog_balance):
    # if type(color_matrix_1[0]) is Ratio:
    #     color_matrix_1 = ratios2floats(color_matrix_1)
    # if type(color_matrix_2[0]) is Ratio:
    #     color_matrix_2 = ratios2floats(color_matrix_2)
    # xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    # xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # # normalize rows (needed?)
    # #xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # #xyz2cam2 = xyz2cam2 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # # inverse
    # cam2xyz1 = np.linalg.inv(xyz2cam1)
    # cam2xyz2 = np.linalg.inv(xyz2cam2)
    # # for now, use one matrix  # TODO: interpolate btween both
    # # simplified matrix multiplication
    # xyz_image = cam2xyz2[np.newaxis, np.newaxis, :, :] * demosaiced_image[:, :, np.newaxis, :]
    # xyz_image = np.sum(xyz_image, axis=-1)
    D_inv = np.array([[as_shot_neutral[0].decimal(), 0, 0],
                      [0, as_shot_neutral[1].decimal(),0],
                      [0, 0, as_shot_neutral[2].decimal()]])
    D = np.linalg.inv(D_inv)


    fm1_elements = []
    for i in forward_matrix_1:
        fm1_elements.append(i.decimal())

    fm2_elements = []
    for i in forward_matrix_2:
        fm2_elements.append(i.decimal())

    cc_elements = []
    for i in color_correction_1:
        cc_elements.append(i.decimal())
    CC = np.array(cc_elements).reshape((3, 3))

    g = 0.5
    FM1 = np.array(fm1_elements).reshape((3,3))
    FM2 = np.array(fm2_elements).reshape((3, 3))
    FM = g*FM1 + (1-g)*FM2

    DF = np.matmul(FM, D)
    DF = np.matmul(DF,np.linalg.inv(CC))


    xyz_image = DF[np.newaxis, np.newaxis, :, :] * demosaiced_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)

    #no clipping
    #xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_srgb(xyz_image):
    # srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
    #                      [0.2126729, 0.7151522, 0.0721750],
    #                      [0.0193339, 0.1191920, 0.9503041]])

    # xyz2srgb = np.linalg.inv(srgb2xyz)

    # xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
    #                      [-0.9692660, 1.8760108, 0.0415560],
    #                      [0.0556434, -0.2040259, 1.0572252]])

    xyz2srgb = np.array([[3.1338561, -1.6168667, -0.4906146],
                         [-0.9787684, 1.9161415, 0.0334540],
                         [0.0719453, -0.2289914, 1.4052427]])

    # xyz2srgb = np.array([[1.3459433, -0.2556075, -0.0511118],
    #                      [-0.5445989, 1.5081673, 0.0205351],
    #                      [0.0000000, 0.0000000, 1.2118128]])

    # normalize rows (needed?)
    #xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    #removing clipping
    #srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reverse_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW
    rev_orientations = np.array([1, 2, 3, 4, 5, 8, 7, 6])
    return fix_orientation(image, rev_orientations[orientation - 1])


def apply_gamma(x):
    x = np.clip(x, 0, 1)
    return np.clip(1.055 * (x ** (1.0 / 2.2)) - 0.055, 0, 1)


def apply_tone_map(x):
    # simple tone curve
    # return 3 * x ** 2 - 2 * x ** 3

    f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'acr3.txt'), "r")

    tones = f.read().replace(' ', '').replace('\t', '').replace('\n', '').split(',')

    inRange = np.linspace(0,1,1025)
    for i, tone in enumerate(tones):
        tones[i] = float(tone)

    # tone_curve = loadmat('tone_curve.mat')
    #tone_curve = loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tone_curve.mat'))
    #tone_curve = tone_curve['tc']
    #x = np.round(x * (len(tone_curve) - 1)).astype(int)
    tone_mapped_image = np.interp(x, inRange, tones)
    return tone_mapped_image


def raw_rgb_to_cct(rawRgb, xyz2cam1, xyz2cam2):
    """Convert raw-RGB triplet to corresponding correlated color temperature (CCT)"""
    pass
    # pxyz = [.5, 1, .5]
    # loss = 1e10
    # k = 1
    # while loss > 1e-4:
    #     cct = XyzToCct(pxyz)
    #     xyz = RawRgbToXyz(rawRgb, cct, xyz2cam1, xyz2cam2)
    #     loss = norm(xyz - pxyz)
    #     pxyz = xyz
    #     fprintf('k = %d, loss = %f\n', [k, loss])
    #     k = k + 1
    # end
    # temp = cct
