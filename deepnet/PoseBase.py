from PoseCommon_dataset import PoseCommon
import PoseTools
import tensorflow as tf
import logging
import sys
import numpy as np
import os

class PoseBase(PoseCommon):
    '''
    Inherit this class to use your own network with APT.
    If the name of your networks is <netname>, then the new class should be created in python file Pose_<netname>.py and the class name should also be Pose_<netname>
    The new class name should be the same

    If the network generates heatmaps for each landmark (i.e., body part), then you only need to override create_network function and supply the appropriate hmaps_downsample to the __init__ function when you inherit.

    If your networks output is different than only heatmaps, you'll have to override
        - preproc_func (to generate the target outputs other than heatmaps).
        - create_network
        - loss
        - convert_preds_to_locs

    In addition, if you want to change the training procedure, you'll have to override
        - train function.
        - get_pred_fn function

    To use pretrained weights, set the self.conf.use_pretrained_weights to True, and put the location of the pretrained weights to self.pretrained_weights. For eg, add the following lines to your __init__ function (udpate resnet_v1_50.ckpt to the appropriate pretrained model):

        self.conf.use_pretrained_weights = True
        script_dir = os.path.dirname(os.path.realpath(__file__))
        wt_dir = os.path.join(script_dir, 'pretrained')
        self.pretrained_weights =  os.path.join(wt_dir,'resnet_v1_50.ckpt')

    '''


    def __init__(self, conf, hmaps_downsample=1):
        '''
        Initialize the pose object.

        :param conf: Configuration object has all the parameters defined in params_netname.yaml. Some important parameters are:
            imsz: 2 element tuple specifying the size of the input image.
            img_dim: Number of channels in input image
            batch_size
            rescale: How much to downsample the input image before feeding into the network.
            dl_steps: Number of steps to run training for.
            In addition, any configuration setting defined in APT_basedir/trackers/dt/params_<netname>.yaml will be available to objects Pose_<netname> in file Pose_<netname> are created.
        :param hmaps_downsample: The amount that the networks heatmaps are downsampled as compared to input image. Ignored if preproc_func is overridden.

        '''

        PoseCommon.__init__(self, conf,name='deepnet')
        self.hmaps_downsample = hmaps_downsample

    def get_var_list(self):
        return tf.global_variables()


    def preproc_func(self, ims, locs, info, distort):
        '''
        Override this function to change how images are preprocessed and to change how labels are converted into heatmaps. Ensure that the return objects are float32.
        This function is added into tensorflow dataset pipeline using tf.py_func. The outputs returned by this function are available tf tensors in self.inputs array.
        You can use PoseTools.create_label_images to generate the target heatmaps.
        You can use PoseTools.create_affinity_labels to generate the target pose affinity field images.
        :param ims: Input image as B x H x W x C
        :param locs: Labeled part locations as B x N x 2
        :param info: Information about the input as B x 3. (:,0) is the movie number, (:,1) is the frame number and (:,2) is the animal number (if the project has trx).
        :param distort: Whether to augment the data or not.
        :return: augmented images, augmented labeled locations, input information, heatmaps.
        '''

        conf = self.conf
        # Scale and augment the training image and labels
        ims, locs = PoseTools.preprocess_ims(ims, locs, conf, distort, conf.rescale)
        hmaps_rescale = self.hmaps_downsample
        hsz = [ (i // conf.rescale)//hmaps_rescale for i in conf.imsz]
        # Creates heatmaps by placing gaussians with sigma label_blur_rad at location locs.
        hmaps = PoseTools.create_label_images(locs/hmaps_rescale, hsz, 1, conf.label_blur_rad)
        # Return the results as float32.
        return ims.astype('float32'), locs.astype('float32'), info.astype('float32'), hmaps.astype('float32')


    def setup_network(self):
        '''
        Setups the network prediction and loss.
        :return:
        '''
        pred = self.create_network()
        self.pred = pred
        self.cost = self.loss(self.inputs,pred)


    def create_network(self):
        '''
        Use self.inputs to create a network.
        If preproc_function is not overridden, then:
            self.inputs[0] tensor has the images [bsz,imsz[0],imsz[1],imgdim]
            self.inputs[1] tensor has the locations in an array of [bsz,npts,2]
            self.inputs[2] tensor has the [movie_id, frame_id, animal_id] information. Mostly useful for debugging.
            self.inputs[3] tensor has the heatmaps, which should ideally not be used for creating the network. Use it only for computing the loss.
        Information about whether it is training phase or test phase for batch norm is available in self.ph['phase_train']
        If preproc function is overridden, then self.inputs will have the outputs of preproc function.
        This function must return the network's output such as the predicted heatmaps.
        '''
        assert False, 'This function must be overridden'
        return None


    def convert_preds_to_locs(self, pred):
        '''
        Converts the networks output to 2D predictions.
        Override this function to write your function to convert the networks output (as numpy array) to locations. Note the locations should be in input images scale i.e., downsampled by self.rescale
        :param pred: Output of network as python/numpy arrays
        :return: 2D locations as batch_size x num_pts x 2
        '''
        return PoseTools.get_pred_locs(pred)*self.hmaps_downsample


    def get_conf_from_preds(self, pred):
        '''
        Computes the prediction confidence.
        :param pred: Output of network as python/numpy arrays
        :return: Confidence for predictions. batch_size x num_pts
        '''
        return np.max(pred, axis=(1, 2))


    def compute_dist(self, pred, locs):
        '''
        Function is used by APT to display the mean pixel error during training
        :param preds: predictions from define_network
        :param locs: label locations as python array
        :return: Mean pixel error between network's prediction and labeled location.
        '''
        tt1 = self.convert_preds_to_locs(pred) - locs
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        return np.nanmean(tt1)


    def loss(self,inputs,pred):
        '''
        :param inputs: Will have the self.inputs tensors return by preproc_func
        :param pred: Has the output of network created in define_network
        :return: The loss function to be optimized.
        Override this define your own loss function.
        '''
        hmap_loss = tf.sqrt(tf.nn.l2_loss(inputs[-1] - self.pred)) / self.conf.label_blur_rad / self.conf.n_classes

        return hmap_loss


    def train_wrapper(self, restore=False):

        # Find the number of outputs from preproc function for the tf.py_func using dummpy inputs, which inserts the preprocessing code into the tensorflow dataset pipeline. The size of output of preproc function is also used to set the size of the tf tensors in self.inputs which will be used during create_network.
        conf = self.conf
        b_sz = conf.batch_size
        imsz = conf.imsz
        img_dim = conf.img_dim
        n_classes = conf.n_classes

        dummy_ims = np.random.rand(b_sz,imsz[0],imsz[1] ,img_dim)
        dummy_locs = np.ones([b_sz,n_classes,2]) * min(imsz)/2
        dummy_info = np.ones([b_sz,3])
        pp_out = self.preproc_func(dummy_ims,dummy_locs,dummy_info,True)
        self.input_dtypes = [tf.float32,]*len(pp_out)

        def train_pp(ims,locs,info):
            return self.preproc_func(ims,locs,info, True)
        def val_pp(ims,locs,info):
            return self.preproc_func(ims,locs,info, False)

        self.train_py_map = lambda ims, locs, info: tuple(tf.py_func( train_pp, [ims, locs, info], self.input_dtypes))
        self.val_py_map = lambda ims, locs, info: tuple(tf.py_func( val_pp, [ims, locs, info], self.input_dtypes ))

        self.setup_train()

        # Set the size for the input tensors
        for ndx, i in enumerate(self.inputs):
            i.set_shape(pp_out[ndx].shape)

        # create the network
        self.setup_network()

        # train
        self.train(restore=restore)

        # reset in case we want to use tensorflow for other stuff.
        tf.reset_default_graph()


    def train(self, restore=False):
        '''
        :param restore: Whether to start training from previously saved model or start from scratch.
        :return:

        This function trains the network my minimizing the loss function using Adam optimizer along with gradient norm clipping.
        The learning rate schedule is exponential decay: lr = conf.learning_rate*(conf.gamma**(float(cur_step)/conf.decay_steps))
        Override this function to implement a different training function. If you override this, do override the get_pred_fn as well.
        The train function should save models every self.conf.save_step to self.conf.cachedir. APT expects the model names to follow format used by tf.train.Saver for saving models (e.g. deepnet-10000.index)
        Before each training step call self.fd_train() which will setup the data generator to generate augmented input images in self.inputs from training database during the next call to sess.run. This also sets the self.ph['phase_train'] to True for batch norm. Use self.fd_val() will generate non-augmented inputs from training DB and to set the self.ph['phase_train'] to false for batch norm.
        To view updated training status in APT, call self.update_and_save_td(step,sess) after each training step. Note update_and_save_td uses the output of loss function to find the loss and convert_preds_to_locs function to find the distance between prediction and labeled locations.
        '''

        base_lr = self.conf.learning_rate
        PoseCommon.train_quick(self, learning_rate=base_lr,restore=restore)


    def get_pred_fn(self, model_file=None):
        '''
        :param model_file: Model_file to use. If not specified the latest trained should be used.
        :return: pred_fn: Function that predicts the 2D locations given a batch of input images.
        :return close_fn: Function to call to close the predictions (eg. the function should close the tensorflow session)
        :return model_file_used: Returns the model file that is used for prediction.

        Creates a prediction function that returns the pose prediction as a python array of size [batch_size,n_pts,2].
        This function should creates the network, start a tensorflow session and load the latest model.

        '''

        try:
            sess, latest_model_file = self.restore_net_common(model_file=model_file)
        except tf.errors.InternalError:
            logging.exception(
                'Could not create a tf session. Probably because the CUDA_VISIBLE_DEVICES is not set properly')
            sys.exit(1)

        conf = self.conf
        def pred_fn(all_f):
            '''
            :param all_f:
            :return:
            This is the function that is used for predicting the location on a batch of images.
            The input is a numpy array B x H x W x C of images, and
            output an numpy array of predicted locations.
            predicted locations should be B x N x 2
            PoseTools.get_pred_locs can be used to convert heatmaps into locations.
            The predicted locations should be in the original image scale.
            The predicted locations, confidence and hmaps(if used) should be returned in a dict with keys 'locs', 'conf' and 'hmaps'.
            If overriding this function, ensure that you call self.fd_val() which will set the self.ph[phase_train] to false for batch norm.
            '''

            bsize = conf.batch_size
            xs, locs_in = PoseTools.preprocess_ims(all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf,
                distort=False, scale=self.conf.rescale)

            self.fd[self.inputs[0]] = xs
            self.fd_val()
            try:
                pred = sess.run(self.pred, self.fd)
            except tf.errors.ResourceExhaustedError:
                logging.exception('Out of GPU Memory. Either reduce the batch size or scale down the images')
                exit(1)
            base_locs = self.convert_preds_to_locs(pred)
            base_locs = base_locs * conf.rescale
            ret_dict = {}
            ret_dict['locs'] = base_locs
            ret_dict['hmaps'] = pred
            ret_dict['conf'] = self.get_conf_from_preds(pred)
            return ret_dict

        def close_fn():
            sess.close()

        return pred_fn, close_fn, latest_model_file