import numpy as np
from .pipeline_utils import get_visible_raw_image, get_metadata, normalize, polynomial, \
    white_balance, demosaic, apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, \
    fix_orientation, lens_shading_correction, vignetting_correction, performInterpolation



def run_pipeline_v2(image_or_path, params=None, metadata=None, fix_orient=True):
    params_ = params.copy()
    if type(image_or_path) == str:
        image_path = image_or_path
        # raw image data
        raw_image = get_visible_raw_image(image_path)
        # metadata
        metadata = get_metadata(image_path)
    else:
        raw_image = image_or_path.copy()
        # must provide metadata
        if metadata is None:
            raise ValueError("Must provide metadata when providing image data in first argument.")

    current_stage = 'raw'
    current_image = raw_image

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        # linearization
        linearization_table = metadata['linearization_table']
        if linearization_table is not None:
            print('Linearization table found. Not handled.')
            # TODO

        current_image = normalize(current_image, metadata['black_level'], metadata['white_level'])
        #current_image = polynomial(current_image, metadata["polynomial"])
        params_['input_stage'] = 'normal'

    current_stage = 'normal'

    if params_['output_stage'] == current_stage:
        return current_image


    if params_['input_stage'] == current_stage:
        current_image = demosaic(current_image, metadata['cfa_pattern'], output_channel_order='RGB',
                                 alg_type=params_['demosaic_type'])
        params_['input_stage'] = 'demosaic'

    current_stage = 'demosaic'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        vignetting_opcode = None
        if 51022 in metadata['opcode_lists']:
            opcode_list_3 = metadata['opcode_lists'][51022]
            if (opcode_list_3 is not None ) and (3 in opcode_list_3):
                vignetting_opcode = opcode_list_3[3]

        if vignetting_opcode is not None:
            print("VIGNETTING")
            current_image = vignetting_correction(current_image, vignetting_opcode = vignetting_opcode)
        params_['input_stage'] = 'vignetting_correction'


    current_stage = 'vignetting_correction'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = apply_color_space_transform(current_image, metadata['camera_calibration_1'],
                                                    metadata['camera_calibration_2'], metadata['forward_matrix_1'], metadata['forward_matrix_2'], metadata['as_shot_neutral'], metadata['analog_balance'])
        params_['input_stage'] = 'xyz'

    current_stage = 'xyz'

    if params_['output_stage'] == current_stage:
        return current_image



    if params_['input_stage'] == current_stage:
        hsv_lut = metadata["hsv_lut"]
        profile_lut = metadata["profile_lut"]
        #comment for now
        # if hsv_lut is not None:
        #     current_image = performInterpolation(current_image, hsv_lut) #hsv sat map

        # if profile_lut is not None:
        #     current_image = performInterpolation(current_image, profile_lut) #profile map

        params_['input_stage'] = 'lut_table'

    current_stage = 'lut_table'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = transform_xyz_to_srgb(current_image)
        params_['input_stage'] = 'srgb'

    current_stage = 'srgb'

    if fix_orient:
        # fix image orientation, if needed (after srgb stage, ok?)
        current_image = fix_orientation(current_image, metadata['orientation'])

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = apply_gamma(current_image)
        params_['input_stage'] = 'gamma'

    current_stage = 'gamma'

    if params_['output_stage'] == current_stage:
        return current_image

    if params_['input_stage'] == current_stage:
        current_image = apply_tone_map(current_image)
        params_['input_stage'] = 'tone'

    current_stage = 'tone'

    if params_['output_stage'] == current_stage:
        return current_image

    # invalid input/output stage!
    raise ValueError('Invalid input/output stage: input_stage = {}, output_stage = {}'.format(params_['input_stage'],
                                                                                              params_['output_stage']))


def run_pipeline(image_path, params):
    # raw image data
    raw_image = get_visible_raw_image(image_path)

    if params['output_stage'] == 'raw':
        return raw_image

    # metadata
    metadata = get_metadata(image_path)

    # linearization
    linearization_table = metadata['linearization_table']
    if linearization_table is not None:
        print('Linearization table found. Not handled.')
        # TODO

    normalized_image = normalize(raw_image, metadata['black_level'], metadata['white_level'])

    if params['output_stage'] == 'normal':
        return normalized_image

    demosaiced_image = demosaic(normalized_image, metadata['cfa_pattern'], output_channel_order='RGB',
                                alg_type=params['demosaic_type'])

    # fix image orientation, if needed
    demosaiced_image = fix_orientation(demosaiced_image, metadata['orientation'])

    if params['output_stage'] == 'demosaic':
        return demosaiced_image

    xyz_image = apply_color_space_transform(demosaiced_image, metadata['color_corection_1'], metadata['color_matrix_2'])

    if params['output_stage'] == 'xyz':
        return xyz_image

    srgb_image = transform_xyz_to_srgb(xyz_image)
    #srgb_image = np.pow(srgb_image, 2.2)
    if params['output_stage'] == 'srgb':
        return srgb_image


    gamma_corrected_image = apply_gamma(srgb_image)
    #gamma_corrected_image = srgb_image
    if params['output_stage'] == 'gamma':
        return gamma_corrected_image

    tone_mapped_image = apply_tone_map(gamma_corrected_image)
    if params['output_stage'] == 'tone':
        return tone_mapped_image

    output_image = None
    return output_image
