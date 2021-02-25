import types
import math
import os
import pickle
import logging
import argparse
import sys
import time
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import APT_interface as apt
import PoseTools as pt
import tfdatagen
import heatmap
import sb1 as sb
import apt_dpk_callbacks

APT_DEEPNET = r'/dat0/git/apt.ma/deepnet'
sys.path.append(APT_DEEPNET)
import movies

def pp(ims, in_locs, distort, scale, mask=None, horz_flip=True, vert_flip=True,
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

    locs = in_locs.copy()
    cur_im = ims.copy()
    cur_im = cur_im.astype('uint8')
    xs = cur_im
    xs, locs, mask = pt.scale_images(xs, locs, scale, conf, mask=mask)  # conf not used

    if distort:
        if horz_flip:
            xs, locs, mask = pt.randomly_flip_lr(xs, locs, conf, group_sz=group_sz, mask=mask)
        if vert_flip:
            xs, locs, mask = pt.randomly_flip_ud(xs, locs, conf, group_sz=group_sz, mask=mask)
        xs = pt.randomly_adjust(xs, conf, group_sz=group_sz)

    xs = xs.astype('float')

    return xs, locs, mask


def ims_locs_pp(imsraw, locsraw, conf, distort, mask=None, gen_target_hmaps=True,):
    '''
    :param imsraw: BHWC
    :param locsraw: B x ntgt x npt x 2
    :param gen_target_hmaps: If false, don't draw hmaps, just return ims, locs
    :return: ims, locs, targets
    '''

    assert conf.imsz == imsraw.shape[1:3]

    do_mask = mask is not None
    if do_mask:
        assert mask.ndim == 3
        assert imsraw.shape[:3] == mask.shape

    # take centroid
    locscent = np.mean(locsraw, axis=2, keepdims=True)

    # TODO: enable padding of mask
    assert conf.sb_im_pady == 0
    assert conf.sb_im_padx == 0
    if False:
        pass
        #imspad, locscentpad = tfdatagen.pad_ims_black(imsraw, locscent, conf.sb_im_pady, conf.sb_im_padx)
    else:
        imspad = imsraw
        locscentpad = locscent
        if do_mask:
            maskpad = mask
    assert imspad.shape[1:3] == conf.sb_imsz_pad
    if do_mask:
        assert maskpad.shape[1:3] == conf.sb_imsz_pad

    ims, locs, mask = pp(imspad, locscentpad, distort, conf.rescale, mask=maskpad,
                         flm=conf.flipLandmarkMatches,
                         brange=conf.brange, crange=conf.crange, imax=conf.imax,
                         )

    imszuse = conf.sb_imsz_net  # network input dims
    (imnr_use, imnc_use) = imszuse
    assert ims.shape[1:3] == imszuse
    assert ims.shape[3] == conf.img_dim
    if do_mask:
        assert mask.shape[1:3] == imszuse

    if conf.img_dim == 1:
        ims = np.tile(ims, 3)

    if not gen_target_hmaps:
        return ims, locs, mask

    assert (imnr_use/conf.sb_output_scale).is_integer() and \
           (imnc_use/conf.sb_output_scale).is_integer(), \
        "Network input size is not even multiple of sb_output_scale"
    imsz_out = [int(x / conf.sb_output_scale) for x in imszuse]
    locs_outres = pt.rescale_points(locs, conf.sb_output_scale, conf.sb_output_scale)
    label_map_outres = heatmap.create_label_hmap(locs_outres,
                                                 imsz_out,
                                                 conf.sb_blur_rad_output_res,
                                                 usefmax=True,
                                                 assert_when_naninf=False,
                                                 )
    targets = label_map_outres


    return ims, locs, targets, mask


def make_data_generator(tfrfilename, conf0, distort, shuffle, silent=False,
                        batch_size=None, **kwargs):
    assert conf0.is_multi
    return tfdatagen.make_data_generator(tfrfilename, conf0, distort, shuffle,
                                         ims_locs_pp, silent=silent, batch_size=batch_size,
                                         **kwargs)

def create_train_callbacks(conf):
    #nowstr = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')

    lr_cbk = apt_dpk_callbacks.create_lr_sched_callback(
        conf.display_step,
        conf.sb_base_lr,
        conf.gamma,
        conf.decay_steps)

    ckpt_reg = 'deepnet-{epoch:08d}.h5'
    ckpt_reg = os.path.join(conf.cachedir, ckpt_reg)
    #ckpt_final = ckpt_reg.format(batch=conf.dl_steps)
    model_checkpoint_reg = tf.keras.callbacks.ModelCheckpoint(
        ckpt_reg,
        save_freq=conf.save_step,  # save every this many batches
        save_best_only=False,
    )

    trn_log = 'trnlog.csv'
    trn_log = os.path.join(conf.cachedir, trn_log)
    logger = tf.keras.callbacks.CSVLogger(trn_log)

    cbks = [lr_cbk, model_checkpoint_reg, logger]
    return cbks


def eucl_loss(x, y):
    return K.sum(K.square(x - y)) / 2.0


def training(conf, return_model=False):

    assert not conf.normalize_img_mean, "SB currently performs its own img input norm"
    assert not conf.normalize_batch_mean, "SB currently performs its own img input norm"

    steps_per_epoch = conf.display_step
    max_epoch = math.ceil(conf.dl_steps/steps_per_epoch)
    assert conf.dl_steps % steps_per_epoch == 0, 'dl steps must be a multiple of display steps'
    assert conf.save_step % steps_per_epoch == 0, ' save steps must be a multiple of display steps'

    train_data_file = os.path.join(conf.cachedir, 'conf.trn.pickle')
    if not return_model:
        with open(train_data_file, 'wb') as td_file:
            pickle.dump(conf, td_file, protocol=2)
        logging.info('Saved config to {}'.format(train_data_file))

    model = sb.get_training_model(conf.sb_imsz_net,
                                  conf.sb_weight_decay_kernel,
                                  nDC=conf.sb_num_deconv,
                                  dc_num_filt=conf.sb_deconv_num_filt,
                                  npts=1,  #conf.n_classes,
                                  backbone=conf.sb_backbone,
                                  backbone_weights=conf.sb_backbone_weights,
                                  mask_strategy=conf.sb_ma_mask_strategy,
                                  )

    conf.sb_output_scale = sb.get_output_scale(model)
    conf.sb_blur_rad_output_res = \
        max(1.0, conf.sb_blur_rad_input_res / float(conf.sb_output_scale))
    logging.info('Model output scale is {}, blurrad_input/output is {}/{}'.format(conf.sb_output_scale, conf.sb_blur_rad_input_res, conf.sb_blur_rad_output_res))

    trntfr = os.path.join(conf.cachedir, conf.trainfilename) + '.tfrecords'
    is_strat2 = conf.sb_ma_mask_strategy == 2
    train_di = make_data_generator(trntfr, conf, True, True, strat2=is_strat2)

    if return_model:
        return model, train_di

    callbacks_list = create_train_callbacks(conf)

    # Epsilon: could just leave un-speced, None leads to default in tf1.14 at least
    # Decay: 0.0 bc lr schedule handled above by callback/LRScheduler
    optimizer = Adam(lr=conf.sb_base_lr, beta_1=0.9, beta_2=0.999,
                     epsilon=None, decay=0.0, amsgrad=False)
    if is_strat2:
        model.compile(loss=None, optimizer=optimizer)
    else:
        model.compile(loss=eucl_loss, optimizer=optimizer)
    logging.info("Your model.metrics_names are {}".format(model.metrics_names))

    # save initial model
    #model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(0))))

    model.fit_generator(train_di,
                        steps_per_epoch=steps_per_epoch,
                        epochs=max_epoch-1,
                        callbacks=callbacks_list,
                        verbose=0,
                        initial_epoch=0,
                        )

    # force saving in case the max iter doesn't match the save step.
    #model.save(str(os.path.join(conf.cachedir, name + '-{}'.format(int(max_epoch*steps_per_epoch)))))
    #obs.on_epoch_end(max_epoch-1)

def train(args,**kwargs):
    if args.data_key == 'ar':
        SLBL = '/dat0/det/dat' + \
               '/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20200317_stripped20200403_new_skl_20200817.lbl'
        conf = apt.create_conf(SLBL, 0, 'centdet', args.run_dir, 'sb', quiet=False)

        conf.img_dim = 1  # hack, the leap stripped lbl has NumChans=3, but we created the tfr
        conf.is_multi = True
        # apt.setup_ma(conf)
        conf.imsz = (1024, 1024)
        conf.rescale = 2
        conf.batch_size = 4
        sb.update_conf(conf)
        conf.max_n_animals = 11
        conf.multi_use_mask = False  # whether mask applied to im at TFR-read-time
        conf.trainfilename = 'train'
        conf.cachedir = args.run_dir
        conf.sb_num_deconv = 0
        conf.sb_output_scale = 8
        conf.sb_blur_rad_output_res = \
            max(1.0, conf.sb_blur_rad_input_res / float(conf.sb_output_scale))
        conf.sb_ma_mask_strategy = args.mask_strat

    return training(conf, **kwargs)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def track(args):

    HMTHRESH = 0.25
    FRMPRINTDEC = 50
    FRMSAVEDEC = 500

    edir = os.path.join(args.base_dir, args.run_name)
    conff = os.path.join(edir, 'conf.trn.pickle')
    mdlf = os.path.join(edir, args.model)
    movf = os.path.join(args.base_dir, args.mov)
    outf = os.path.join(edir, args.out)

    conf = pt.pickle_load(conff)
    print("Loaded conf from {}".format(conff))
    assert not conf.normalize_img_mean, "SB currently performs its own img input norm"
    assert not conf.normalize_batch_mean, "SB currently performs its own img input norm"

    m = movies.Movie(movf)
    im0, _ = m.get_frame(0)
    imsz = im0.shape
    print("Movie image size is: {}".format(imsz))

    model = sb.get_training_model(imsz,
                                  None,
                                  nDC=conf.sb_num_deconv,
                                  dc_num_filt=conf.sb_deconv_num_filt,
                                  npts=1,
                                  backbone=conf.sb_backbone,
                                  backbone_weights=None,
                                  upsamp_chan_handling=conf.sb_upsamp_chan_handling)
    model.load_weights(mdlf)
    print("Loaded model weights from {}".format(mdlf))

    def savefcn(stuff):
        with open(outf, 'w') as fh:
            json.dump(stuff, fh, cls=NpEncoder)
        print("Wrote {} rows to {}".format(len(stuff), outf))

    f0 = args.f0
    f1 = args.f1
    pks = []
    start = time.time()
    for frm in range(f0, f1):
        im, ts = m.get_frame(frm)
        im3 = np.dstack((im, im, im))
        im3 = im3[np.newaxis, ...]
        hm = model.predict(im3)
        assert hm.shape[0] == 1
        assert hm.shape[-1] == 1
        hm = hm[0, ..., 0]
        pksthis = heatmap.find_peaks_nlargest(hm, 5)
        #if len(pksthis) != 2:
        #    print("{}: len(pks)={}".format(frm, len(pksthis)))
        pks.append(pksthis)

        if frm % FRMPRINTDEC == 0:
            print("predicted frm {}".format(frm))

        if frm % FRMSAVEDEC == 0:
            savefcn(pks)

    end = time.time()
    print("Elapsed time is: {}".format(end-start))

    savefcn(pks)



def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_dir', default="/dat0/det", help="All other dirs/files spec'd wrt this dir")
    parser.add_argument("-run_name", required=True)
    parser.add_argument("-data_key", choices=['ar', 'nor'], help="convenience arg")
    #parser.add_argument('-data_dir')
    #parser.add_argument('-data_ann')
    #parser.add_argument('-cfg_yaml', required=True)
    subparsers = parser.add_subparsers(help='train or track', dest='action')
    parser_train = subparsers.add_parser('train', help='Train the detector')
    parser_train.add_argument('-mask_strat', choices=[0,1,2], type=int, required=True)
    parser_track = subparsers.add_parser('track', help='Track a movie')
    parser_track.add_argument('-model', required=True, help='short/relative model filename')
    parser_track.add_argument("-mov", required=True, help="movie to track")
    parser_track.add_argument("-out", default="trk.json", help="movie to track")
    parser_track.add_argument('-f0', help='start tracking from this frame', type=int, default=1)
    parser_track.add_argument('-f1', help='end frame for tracking', type=int, default=-1)
    # track outdir; for now, use run_dir
    print(argv)

    args = parser.parse_args(argv)
    args.run_dir = os.path.join(args.base_dir, args.run_name)
    assert os.path.exists(args.run_dir)
    #if args.data_key is not None:
    #    assert args.data_dir is None and args.data_ann is None, "If data_key is supplied, don't supply data_dir or data_ann."
    #    if args.data_key == 'ar':
    #        args.data_dir = 'dat/ar_maskim_split_fullims'
    #        args.data_ann = 'dat/ar_maskim_split_fullims/trn.json'
    #    elif args.data_key == 'nor':
    #        args.data_dir = 'dat/fullims'
    #        args.data_ann = 'dat/fullims/trn.json'

    #if args.action == 'train':
    #    args.data_dir = os.path.join(args.base_dir, args.data_dir)
    #    args.data_ann_orig = args.data_ann
    #    args.data_ann = os.path.join(args.base_dir, args.data_ann)
    #if args.action == 'track':
    #    args.data_dir = None
    #    args.data_ann_orig = None
    #    args.data_ann = None
    #    args.cfg_pred_yaml = os.path.join(args.run_dir, args.pred_cfg_yaml) if args.pred_cfg_yaml else None
    #else:
    #    assert False, "Unrecognized action"

    args.cfg_yaml = os.path.join(args.run_dir, 'cfg.yaml')
    args.cfg_full_yaml = os.path.join(args.run_dir, 'cfg_full.yaml')

    return args

def main(argv):
    args = parse_args(argv)
    fcn = globals()[args.action]
    fcn(args)


if __name__ == "__main__":
    print(os.path.abspath(__file__))
    main(sys.argv[1:])