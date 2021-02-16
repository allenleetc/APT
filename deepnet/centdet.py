import types

import numpy as np

import PoseTools as pt
import tfdatagen
import heatmap

def pp(ims, in_locs, distort, horz_flip=True, vert_flip=True,
       flm=None, brange=None, crange=None, imax=None):
    '''

    :param ims: Input image. It is converted to uint8 before applying the transformations. Size: B x H x W x C
    :param in_locs: B x ntgt x npt x 2
    :return:
    '''
#    assert ims.dtype == 'uint8', 'Preprocessing only work on uint8 images'

    conf = types.SimpleNamespace()
    conf.flipLandmarkMatches = flm
    conf.brange = brange
    conf.crange = crange
    conf.imax = imax
    group_sz = 1
    mask = None

    locs = in_locs.copy()
    cur_im = ims.copy()
    cur_im = cur_im.astype('uint8')
    xs = cur_im
    if distort:
        if horz_flip:
            xs, locs, mask = pt.randomly_flip_lr(xs, locs, conf, group_sz=group_sz, mask=mask)
        if vert_flip:
            xs, locs, mask = pt.randomly_flip_ud(xs, locs, conf, group_sz=group_sz, mask=mask)
        xs = pt.randomly_adjust(xs, conf, group_sz=group_sz)

    xs = xs.astype('float')

    return xs, locs


def ims_locs_pp(imsraw, locsraw, conf, distort, gen_target_hmaps=True, mask=None):
    '''
    :param imsraw: BHWC
    :param locsraw: B x ntgt x npt x 2
    :param gen_target_hmaps: If false, don't draw hmaps, just return ims, locs
    :return: ims, locs, targets
    '''

    assert conf.imsz == imsraw.shape[1:3]

    # take centroid
    locsraw = np.mean(locsraw, axis=2)
    locsraw = locsraw[:, :, np.newaxis, :]

    imspad, locspad = tfdatagen.pad_ims_black(imsraw, locsraw, conf.sb_im_pady, conf.sb_im_padx)
    assert imspad.shape[1:3] == conf.sb_imsz_pad

    ims, locs = pp(imspad, locspad, distort,
                   flm=conf.flipLandmarkMatches,
                   brange=conf.brange, crange=conf.crange, imax=conf.imax,
                   )

    imszuse = conf.sb_imsz_net  # network input dims
    (imnr_use, imnc_use) = imszuse
    assert ims.shape[1:3] == imszuse
    assert ims.shape[3] == conf.img_dim
    if conf.img_dim == 1:
        ims = np.tile(ims, 3)

    if not gen_target_hmaps:
        return ims, locs

    assert (imnr_use/conf.sb_output_scale).is_integer() and \
           (imnc_use/conf.sb_output_scale).is_integer(), \
        "Network input size is not even multiple of sb_output_scale"
    imsz_out = [int(x / conf.sb_output_scale) for x in imszuse]
    locs_outres = pt.rescale_points(locs, conf.sb_output_scale, conf.sb_output_scale)
    label_map_outres = heatmap.create_label_hmap(locs_outres,
                                                 imsz_out,
                                                 conf.sb_blur_rad_output_res,
                                                 usefmax=True)
    targets = [label_map_outres,]

#    if not __ims_locs_preprocess_sb_has_run__:
#        logr.info('sb preprocess. sb_out_scale={}, imszuse={}, imszout={}, blurradout={#}'.format(conf.sb_output_scale, imszuse, imsz_out, conf.sb_blur_rad_output_res))
 #       __ims_locs_preprocess_sb_has_run__ = True

    return ims, locs, targets, None


def make_data_generator(tfrfilename, conf0, distort, shuffle, silent=False,
                        batch_size=None, **kwargs):
    assert conf0.is_multi
    return tfdatagen.make_data_generator(tfrfilename, conf0, distort, shuffle,
                                         ims_locs_pp, silent=silent, batch_size=batch_size,
                                         **kwargs)