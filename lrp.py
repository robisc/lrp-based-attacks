import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from keras import backend as K

class LrpExplainer:

    def __init__(self, model, process, a, b, verbose):
        self.process = process
        self.model = model
        self.a = a
        self.b = b
        self.verbose = verbose
        self.eps = 1e-16

    def check_bool(self, x):
        """returns true, if either a single bool variable or an array of bools contains any true value 

        Args:
            x (_type_): list or bool

        Returns:
            bool: True or False
        """
        if type(x) == np.ndarray:
            return x.any()
        else:
            return x

    def div0(self, a, b):
        #""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( a, b )
            c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c

    def relu(self, matrix):
        # Relu implementation returning 0 or the maximum value of a matrix
        return np.maximum(0,matrix)

    def get_outputs(self, image, model):
        # Based on the input data, this function returns all layers' outputs
        outputs = []
        for i in range(0,len(model.layers)):
            layer_output = K.function([model.layers[0].input],
                                        [model.layers[i].output])
            outputs.append(layer_output([image.reshape(tuple([1]+list(image.shape)))])[0])
        return outputs

    def get_inputs(self, image, model):
        # Based on the input data, this function returns all layers' inputs
        inputs = []
        for i in range(0,len(model.layers)):
            layer_input = K.function([model.layers[0].input],
                                        [model.layers[i].input])
            inputs.append(layer_input([image.reshape(tuple([1]+list(image.shape)))])[0])
        return inputs

    def get_weights(self, model):
        # Based on the input data, this function returns all layers' weights
        weights = []
        for i in range(0,len(model.layers)):
            try:
                weights.append(model.layers[i].get_weights()[0])
            except: 
                weights.append(None)
        return weights

    def get_biases(self, model):
        # Based on the input data, this function returns all layers' biases
        biases = []
        for i in range(0,len(model.layers)):
            try:
                biases.append(model.layers[i].get_weights()[1])
            except: 
                biases.append(None)
        return biases

    # Layer specific LRP Rules
    # The choice what rule is used per layer is done in the LRP process
    # For the basic lrp-rules, https://git.tu-berlin.de/gmontavon/lrp-tutorial was used as inspiration

    def relprop_lin_0(self, layer, R, inputs, weights, biases):
        Z = np.matmul(inputs[layer],weights[layer]) + biases[layer]
        S = self.div0(R,Z)
        C = np.matmul(S,weights[layer].T)
        R = C*inputs[layer]
        return R

    def relprop_lin_eps(self, layer, R, inputs, weights, biases): 
        Z = np.matmul(inputs[layer],weights[layer]) + biases[layer] + self.eps 
        S = self.div0(R,Z)
        C = np.matmul(S,weights[layer].T)
        R = C*inputs[layer]
        return R

    def relprop_lin_ab(self, layer, R, inputs, outputs, weights, biases, a, b):
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        mask_p = weights[layer].copy()
        mask_p = np.array(list(map(lambda x: (inputs[layer] * x)>=0,mask_p.T))).T.reshape(weights[layer].shape)
        mask_n = weights[layer].copy()
        mask_n = np.array(list(map(lambda x: (inputs[layer] * x)<0,mask_n.T))).T.reshape(weights[layer].shape)
        Z_p = np.matmul(inputs[layer],weights[layer]*mask_p)+biases[layer]*(biases[layer]>=0)
        Z_p = Z_p + (np.ones(Z_p.shape)*self.eps)*((Z_p)>0)
        Z_n = np.matmul(inputs[layer],weights[layer]*mask_n)+biases[layer]*(biases[layer]<0)
        Z_n = Z_n - (np.ones(Z_n.shape)*self.eps)*((Z_n)<0)
        S_p = self.div0(R,Z_p)
        S_n = self.div0(R,Z_n)
        C_p = np.matmul(S_p,(weights[layer]*mask_p).T)
        C_n = np.matmul(S_n,(weights[layer]*mask_n).T)
        R = (C_p*inputs[layer]*a+C_n*inputs[layer]*b) 
        return R

    def relprop_flatten(self, layer,R, inputs):
        return R.reshape(inputs[layer].shape)

    def relprop_pooling_avg(self, layer, R, inputs, outputs, model):
        pool_height, pool_width = model.layers[layer].get_config()["pool_size"]
        stride_up, stride_side = model.layers[layer].get_config()["strides"]
        placeholder = np.zeros(inputs[layer].shape)
        S = R
        for l in range(outputs[layer].shape[3]):
            for i in range(outputs[layer].shape[2]):
                for j in range(outputs[layer].shape[1]):
                    placeholder[0,i*stride_side:(i*stride_side+pool_width),j*stride_up:(j*stride_up+pool_height),l] = S[0,i,j,l]*(np.ones(placeholder[0,i*stride_side:(i*stride_side+pool_width),j*stride_up:(j*stride_up+pool_height),l].shape))*1/(pool_height * pool_width)
        R = placeholder
        return R
        
    def relprop_pooling(self, layer, R, inputs, outputs, model):
        pool_height, pool_width = model.layers[layer].get_config()["pool_size"]
        stride_up, stride_side = model.layers[layer].get_config()["strides"]
        placeholder = np.zeros(inputs[layer].shape)
        S = R
        for l in range(outputs[layer].shape[3]):
            for i in range(outputs[layer].shape[2]):
                for j in range(outputs[layer].shape[1]):
                    placeholder[0,i*stride_side:(i*stride_side+pool_width),j*stride_up:(j*stride_up+pool_height),l] = \
                    S[0,i,j,l]*((inputs[layer][0,i*stride_side:(i*stride_side+pool_width),j*stride_up:(j*stride_up+pool_height),l] == \
                    outputs[layer][0,i,j,l])&(outputs[layer][0,i,j,l]!=0))
        R = placeholder
        return R

    def relprop_pooling1(self, layer, R, inputs, outputs, model):
        # Returns the identical values as relprop_pooling, however works faster due to utilizing Keras functions
        pool_s = tuple([1]+list(model.layers[layer].get_config()["pool_size"])+[1])
        padding = model.layers[layer].get_config()["padding"]
        placeholder = K.eval(gen_nn_ops.max_pool_grad_v2(inputs[layer], outputs[layer], R, pool_s, pool_s, padding=padding.upper()))
        R = placeholder
        return R

    def relprop_pooling1_avg(self, layer, R, inputs, outputs, model):
        # Returns the identical values as relprop_pooling_avg, however works faster due to utilizing Keras functions
        pool_s = tuple([1]+list(model.layers[layer].get_config()["pool_size"])+[1])
        padding = model.layers[layer].get_config()["padding"]
        placeholder = K.eval(gen_nn_ops.avg_pool_grad(inputs[layer].shape, R, pool_s, pool_s, padding=padding.upper()))
        R = placeholder
        return R

    def relprop_conv2d_eps(self, layer, R, inputs, weights, biases, model):
        padding = model.layers[layer].get_config()["padding"]
        Z = K.eval(K.conv2d(tf.constant(inputs[layer]),tf.constant(weights[layer]),strides=(1,1), padding=padding))
        Z = Z + biases[layer]*(Z!=0)
        Z = Z + (np.ones(Z.shape)*self.eps)*(Z!=0)
        S = self.div0(R,Z)
        C = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape, weights[layer],S , (1,1,1,1),padding=padding.upper()))
        R = C*inputs[layer]
        return R

    def relprop_conv2d_first(self, layer, R, inputs, weights, biases, model):
        X = inputs[layer]
        L = inputs[layer]*0 + -1 #-1
        H = inputs[layer]*0 + 1
        W_pos = np.maximum(0,weights[layer])
        W_neg = np.minimum(0,weights[layer])
        padding = model.layers[layer].get_config()["padding"]
        Z = K.eval(K.conv2d(tf.constant(inputs[layer]),tf.constant(weights[layer]),strides=(1,1), padding=padding))
        Z = Z - K.eval(K.conv2d(tf.constant(L),tf.constant(W_pos),strides=(1,1), padding=padding))
        Z = Z - K.eval(K.conv2d(tf.constant(H),tf.constant(W_neg),strides=(1,1), padding=padding))
        S = self.div0(R,Z)
        
        C = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape, weights[layer],S, (1,1,1,1),padding=padding.upper() ))
        C_p = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape,W_pos,S, (1,1,1,1),padding=padding.upper() ))
        C_n = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape,W_neg,S, (1,1,1,1),padding=padding.upper() ))
        R = C*inputs[layer] - C_p*L - C_n*H
        return R

    def relprop_conv2d_ab(self, layer, R, inputs, weights, biases, model, a, b):
        padding = model.layers[layer].get_config()["padding"]

        Z_pp = K.eval(K.conv2d(tf.constant(inputs[layer] * (inputs[layer]>=0)),tf.constant(weights[layer] * (weights[layer]>=0)),strides=(1,1), padding=padding))
        Z_pn = K.eval(K.conv2d(tf.constant(inputs[layer] * (inputs[layer]<0)),tf.constant(weights[layer] * (weights[layer]<0)),strides=(1,1), padding=padding))

        Z_p = Z_pp + Z_pn

        Z_p = Z_p + biases[layer]*(biases[layer]>=0)*(Z_p>=0)
        Z_p = Z_p + (np.ones(Z_p.shape)*self.eps)*((Z_p)>=0)
        Z_npn = K.eval(K.conv2d(tf.constant(inputs[layer] * (inputs[layer]>=0)),tf.constant(weights[layer] * (weights[layer]<0)),strides=(1,1), padding=padding))
        Z_nnp = K.eval(K.conv2d(tf.constant(inputs[layer] * (inputs[layer]<0)),tf.constant(weights[layer] * (weights[layer]>=0)),strides=(1,1), padding=padding))
        Z_n = Z_npn + Z_nnp

        Z_n = Z_n + biases[layer]*(biases[layer]<0)*(Z_n<0)
        Z_n = Z_n - (np.ones(Z_n.shape)*self.eps)*((Z_n)<0)

        S_pp = self.div0(R,Z_pp)
        S_pn = self.div0(R,Z_pn)
        S_npn = self.div0(R,Z_npn)
        S_nnp = self.div0(R,Z_nnp)

        C_pp = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape,weights[layer]*(weights[layer]>=0),S_pp, (1,1,1,1),padding=padding.upper() ))*(inputs[layer]*(inputs[layer]>=0))
        C_pn = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape,weights[layer]*(weights[layer]<0),S_pn, (1,1,1,1),padding=padding.upper() ))*(inputs[layer]*(inputs[layer]<0))

        C_npn = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape,weights[layer]*(weights[layer]<0),S_npn, (1,1,1,1),padding=padding.upper() ))*(inputs[layer]*(inputs[layer]>=0))
        C_nnp = K.eval(tf.compat.v1.nn.conv2d_backprop_input(inputs[layer].shape,weights[layer]*(weights[layer]>=0),S_nnp, (1,1,1,1),padding=padding.upper() ))*(inputs[layer]*(inputs[layer]<0))

        C_p = C_pp + C_pn
        C_n = C_npn + C_nnp

        R = (C_p*a+C_n*b)
        return R

    def relprop_batch_norm(self, layer, R, inputs, outputs, model):
        weights = model.layers[layer].get_weights()
        gamma = weights[0]
        beta = weights[1]
        mean = weights[2]
        std = weights[3]
        s = gamma/np.sqrt(std**2 + 0.001)
        x_dash = inputs[layer]-mean
        x_dash2 = x_dash * s
        z = x_dash2 + beta
        return inputs[layer] * s * R / z

    # Relevance propagation process from here on out

    def relprop(self, img, R):
        if self.verbose: 
            print("calculating LRP of ",str(self.model))
            if self.process: print(self.process)
            print("###################")
            print("getting values")
        inputs = self.get_inputs(img, self.model)
        outputs = self.get_outputs(img, self.model)
        weights = self.get_weights(self.model)
        biases = self.get_biases(self.model)
        rs = []
        if self.verbose: print("propagating relevance regarding classification: ", np.argmax(R))
        
        for i in range(-1,-len(self.model.layers),-1):
            if (i-1 == -len(self.model.layers)):
                if not self.process:
                    R = self.relprop_conv2d_first(i, R, inputs, weights, biases, self.model) 
                else:
                    if self.process[i] == "final":
                        R = self.relprop_conv2d_first(i, R, inputs, weights, biases, self.model)
                    elif self.process[i] == "ab":
                        R = self.relprop_conv2d_ab(i, R, inputs, weights, biases, self.model, self.a, self.b)
                if self.verbose: print("In first layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
                rs.append(R)
                break

            elif "Dense" in str(self.model.layers[i]):
                if not self.process:
                    if self.verbose: print("Default ab Rule for dense used")
                    R = self.relprop_lin_ab(i, R, inputs, outputs, weights, biases, self.a, self.b) #ab
                else:
                    if self.process[i] == "eps":
                        R = self.relprop_lin_eps(i, R, inputs, weights, biases)
                    elif self.process[i] == "0":
                        R = self.relprop_lin_0(i, R, inputs, weights, biases)
                    else:
                        R = self.relprop_lin_ab(i, R, inputs, outputs, weights, biases, self.a, self.b) #ab
                if self.verbose: print("In layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
            elif "Flatten" in str(self.model.layers[i]):
                R = self.relprop_flatten(i , R, inputs)
                if self.verbose: print("In layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
            elif "MaxPool" in str(self.model.layers[i]):
                if not self.process:
                    R = self.relprop_pooling1(i ,R , inputs, outputs, self.model)
                else:
                    if self.process[i] == "avg":
                        R = self.relprop_pooling1_avg(i ,R , inputs, outputs, self.model)
                    else:
                        R = self.relprop_pooling1(i ,R , inputs, outputs, self.model)
                if self.verbose: print("In layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
            elif "AveragePool" in str(self.model.layers[i]):
                R = self.relprop_pooling1_avg(i ,R , inputs, outputs, self.model)
                if self.verbose: print("In layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
            elif "Conv2D" in str(self.model.layers[i]):
                if not self.process:
                    R = self.relprop_conv2d_ab(i, R, inputs, weights, biases, self.model, self.a, self.b)
                else:
                    
                    if self.process[i] == "eps":
                        R = self.relprop_conv2d_eps(i, R, inputs, weights, biases, self.model)
                    elif self.process[i] == "wpn":
                        R = self.relprop_conv2d_wpn(i, R, inputs, weights, biases, self.model, self.a, self.b)
                    else:
                        R = self.relprop_conv2d_ab(i, R, inputs, weights, biases, self.model, self.a, self.b)
                if self.verbose: print("In layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
            elif "Dropout" in str(self.model.layers[i]):
                if self.verbose: print("In layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
            elif "BatchNormalization" in str(self.model.layers[i]):
                # R = relprop_batch_norm(i, R, inputs, outputs, model)
                if self.verbose: print("In layer ",i," : ",self.model.layers[i]," check-value: ", np.sum(R))
            rs.append(R)
        rs.reverse()

        if self.verbose:
            return R, inputs, outputs, weights, biases, rs
        else:
            return R

    # attack functions

    def flip_attack(self, img, label, flips, log=False, early_stopping = False):
        if log:
            y_log = []
            y_log.append(self.model.predict(img.reshape([1]+list(img.shape)),verbose = 0)[0][np.argmax(label)])
            y = self.model.predict(img.reshape([1]+list(img.shape)),verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
            r_log = []
            img_log = []
        flipped_img = img.copy()
        for j in range(0,flips):
            if self.verbose:
                R, inputs, outputs, weights, biases, rs = self.relprop(flipped_img, label)
            else:
                R = self.relprop(flipped_img, label)
            flipping_mask = R==5
            flipping_list = np.sort(R, axis=None)[:]
            if flipping_list[-1]>0:
                flipping_mask[R==flipping_list[-1]]=True
            flipped_img[flipping_mask[0,:,:,:]]= flipped_img[flipping_mask[0,:,:,:]]*-1
            if log: 
                y_log.append(self.model.predict(flipped_img.reshape([1]+list(img.shape)),verbose = 0)[0])
                r_log.append(R)
                temp = flipped_img.copy()
                img_log.append(temp)
            if early_stopping:
                if log:
                    prediction = y_log[-1]
                else:
                    prediction = self.model.predict(flipped_img.reshape([1]+list(img.shape)),verbose = 0)[0]
                if np.argmax(prediction) != np.argmax(label):
                    if log:
                        y_new = prediction
                        y_log = y_log+y_log[-1:]*(flips+1-len(y_log))
                        r_log = r_log+r_log[-1:]*(flips-len(r_log))
                        img_log = img_log+img_log[-1:]*(flips-len(img_log))
                        return flipped_img, y, y_new, y_log, r_log, img_log
                    else:
                        return flipped_img
        if log:
            y_new = self.model.predict(flipped_img.reshape([1]+list(img.shape)),verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
            return flipped_img, y, y_new, y_log, r_log, img_log
        else:
            return flipped_img

    def flip_attack_batch(self, img, label, flips, batch, log=False, early_stopping = False):
        if log:
            y_log = []
            y_log.append(self.model.predict(img.reshape([1]+list(img.shape)),verbose = 0)[0])
            y = self.model.predict(img.reshape([1]+list(img.shape)),verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
            r_log = []
            img_log = []
        flipped_img = img.copy()
        for j in range(0,flips):
            if self.verbose:
                R, inputs, outputs, weights, biases, rs = self.relprop(flipped_img, label)
            else:
                R = self.relprop(flipped_img, label)
            flipping_mask = R==5
            flipping_list = np.sort(R, axis=None)[:]
            for i in range(1,batch+1):
                if flipping_list[-i]>0:
                    flipping_mask[R==flipping_list[-i]]=True
            flipped_img[flipping_mask[0,:,:,:]]= flipped_img[flipping_mask[0,:,:,:]]*-1
            if log: 
                y_log.append(self.model.predict(flipped_img.reshape([1]+list(img.shape)),verbose = 0)[0])
                r_log.append(R)
                temp = flipped_img.copy()
                img_log.append(temp)
            if early_stopping:
                if log:
                    prediction = y_log[-1]
                else:
                    prediction = self.model.predict(flipped_img.reshape([1]+list(img.shape)),verbose = 0)[0]
                if np.argmax(prediction) != np.argmax(label):
                    if log:
                        y_new = prediction
                        return flipped_img, y, y_new, y_log, r_log, img_log
                    else:
                        return flipped_img
        if log: y_new = self.model.predict(flipped_img.reshape([1]+list(img.shape)),verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
        if log:
            return flipped_img, y, y_new, y_log, r_log, img_log
        else:
            return flipped_img

    def lrp_attack(self, img, label, flips, batch, eps, log=False, early_stopping = False):
        if log:
            y_log = []
            y_log.append(self.model.predict(img.reshape([1]+list(img.shape)),verbose = 0)[0])
            y = self.model.predict(img.reshape([1]+list(img.shape)),verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
            r_log = []
            img_log = []
        flipped_img = img.copy()
        for j in range(0,flips):
            if self.verbose:
                R, inputs, outputs, weights, biases, rs = self.relprop(flipped_img, label)
            else:
                R = self.relprop(flipped_img, label)
            flipping_mask = R==5
            flipping_list = np.sort(R, axis=None)[:]
            for i in range(1,batch+1):
                if flipping_list[-i]>0:
                    flipping_mask[R==flipping_list[-i]]=True
                    flipping_mask_one = R==5
                    flipping_mask_one[R==flipping_list[-i]]=True
                    flipped_img[flipping_mask[0,:,:,:]]= flipped_img[flipping_mask[0,:,:,:]]-np.sign(flipped_img[flipping_mask[0,:,:,:]])*eps
            if log: 
                y_log.append(self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0])
                r_log.append(R)
                temp = flipped_img.copy()
                img_log.append(temp)
            if early_stopping:
                if log:
                    prediction = y_log[-1]
                else:
                    prediction = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0]
                if np.argmax(prediction) != np.argmax(label):
                    if log:
                        y_new = prediction
                        return flipped_img, y, y_new, y_log, r_log, img_log
                    else:
                        return flipped_img
        if log: y_new = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
        if log:
            return flipped_img, y, y_new, y_log, r_log, img_log
        else:
            return flipped_img

    def flip_attack_targeted(self, img, target, flips, log=False):
        if log:
            y_log = []
            y_log.append(self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[0])
            y = self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[target.astype(bool).reshape(1,len(target))][0]
            r_log = []
            img_log = []
        flipped_img = img.copy()
        for j in range(0,flips):
            if self.verbose:
                R, inputs, outputs, weights, biases, rs = self.relprop(flipped_img, target)
            else:
                R = self.relprop(flipped_img, target)
            flipping_mask = R==5
            flipping_list = np.sort(R, axis=None)[-1::-1]
            if flipping_list[-1]<0:
                flipping_mask[R==flipping_list[-1]]=True
            flipped_img[flipping_mask[0,:,:,:]]= flipped_img[flipping_mask[0,:,:,:]]*-1
            if log:
                y_log.append(self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0])
                r_log.append(R)
                temp = flipped_img.copy()
                img_log.append(temp)
        if log: y_new = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[target.astype(bool).reshape(1,len(target))][0]
        if log:
            return flipped_img, y, y_new, y_log, r_log, img_log
        else:
            return flipped_img

    def create_adversarial_pattern(self, input_image, input_label):
        # Codebase from https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        input_image = tf.convert_to_tensor(input_image.reshape([1]+list(input_image.shape)))
        with tf.GradientTape() as tape:
            tape.watch(input_image)        
            prediction = self.model(input_image)
            loss = loss_object(input_label.reshape(1,-1), prediction)
        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient)
        return signed_grad, gradient

    def lrp_attack_batch_grad(self, img, label, flips, batch, eps, log=False, early_stopping = False):
        if log:
            y_log = []
            y_log.append(self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[0])
            y = self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
            r_log = []
            img_log = []
        flipped_img = img.copy()
        pattern, _ = self.create_adversarial_pattern(img, label)
        for j in range(0,flips):
            pattern, _ = self.create_adversarial_pattern(flipped_img, label)
            if self.verbose:
                R, inputs, outputs, weights, biases, rs = self.relprop(flipped_img, label)
            else:
                R = self.relprop(flipped_img, label)
            flipping_mask = R==5
            flipping_list = np.sort(R, axis=None)[:]
            i = 1
            limit = batch+1
            while i < limit:
                if flipping_list[-i]>0:
                    if self.check_bool((flipped_img[R[0,:,:,:]==flipping_list[-i]] < -1)) or self.check_bool((flipped_img[R[0,:,:,:]==flipping_list[-i]] > 1)):
                        i = i+1
                        limit = limit+1
                        next
                    flipping_mask[R==flipping_list[-i]]=True
                i = i+1  
            flipped_img[flipping_mask[0,:,:,:]]= flipped_img[flipping_mask[0,:,:,:]]+pattern[flipping_mask[:,:,:]]*eps
            flipping_mask = R==5
            i = 1
            limit = batch+1
            while i < limit:
                if flipping_list[i]<0:
                    if self.check_bool((flipped_img[R[0,:,:,:]==flipping_list[i]] < -1)) or self.check_bool((flipped_img[R[0,:,:,:]==flipping_list[i]] > 1)):
                        i = i+1
                        limit = limit+1
                        next
                    flipping_mask[R==flipping_list[i]]=True
                i = i+1
            flipped_img[flipping_mask[0,:,:,:]]= flipped_img[flipping_mask[0,:,:,:]]+pattern[flipping_mask[:,:,:]]*eps
            flipped_img = np.clip(flipped_img, -1, 1)
            if log: 
                y_log.append(self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0])
                r_log.append(R)
                temp = flipped_img.copy()
                img_log.append(temp)
            if early_stopping:
                if log:
                    prediction = y_log[-1]
                else:
                    prediction = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0]
                if np.argmax(prediction) != np.argmax(label):
                    if log:
                        y_new = prediction
                        return flipped_img, y, y_new, y_log, r_log, img_log
                    else:
                        return flipped_img
        if log: y_new = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
        if log:
            return flipped_img, y, y_new, y_log, r_log, img_log
        else:
            return flipped_img

    def ifgsm_attack(self, img, label, flips, eps, log=False, early_stopping = False):
        if log:
            y_log = []
            y_log.append(self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[0])
            y = self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
            r_log = []
            img_log = []
        flipped_img = img.copy()
        pattern, _ = self.create_adversarial_pattern(img, label)
        for j in range(0,flips):
            pattern, _ = self.create_adversarial_pattern(flipped_img, label)
            pattern = pattern[0,:,:,:]
            flipped_img = flipped_img + pattern*eps
            flipped_img = np.clip(flipped_img, -1, 1)
            if log: 
                y_log.append(self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0])
                temp = flipped_img.copy()
                img_log.append(temp)
                r_log.append(_.numpy())
            if early_stopping:
                if log:
                    prediction = y_log[-1]
                else:
                    prediction = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0]
                if np.argmax(prediction) != np.argmax(label):
                    if log:
                        y_new = prediction
                        return flipped_img, y, y_new, y_log, r_log, img_log
                    else:
                        return flipped_img
        if log: y_new = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
        if log:
            return flipped_img, y, y_new, y_log, r_log, img_log
        else:
            return flipped_img

    def lrp_attack_mean_batch(self, img, label, flips, batch, eps, log=False, early_stopping = False):
        if log:
            y_log = []
            y_log.append(self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[0])
            y = self.model.predict(img.reshape([1]+list(img.shape)), verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
            r_log = []
            img_log = []
        flipped_img = img.copy()
        means = img.mean(axis = (0,1))
        for j in range(0,flips):
            if self.verbose:
                R, inputs, outputs, weights, biases, rs = self.relprop(flipped_img, label)
            else:
                R = self.relprop(flipped_img, label)
            flipping_mask = R==5
            flipping_list = np.sort(R, axis=None)[:]
            for i in range(1,batch+1):
                if flipping_list[-i]>0:
                    flipping_mask[R==flipping_list[-i]]=True
            for i in range(0,img.shape[-1]):
                flipped_img[flipping_mask[0,:,:,i]]= flipped_img[flipping_mask[0,:,:,i]] + np.sign(means[i] - flipped_img[flipping_mask[0,:,:,i]])*eps
            flipped_img = np.clip(flipped_img, -1, 1)
            if log: 
                y_log.append(self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0])
                r_log.append(R)
                temp = flipped_img.copy()
                img_log.append(temp)
            if early_stopping:
                if log:
                    prediction = y_log[-1]
                else:
                    prediction = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[0]
                if np.argmax(prediction) != np.argmax(label):
                    if log:
                        y_new = prediction
                        return flipped_img, y, y_new, y_log, r_log, img_log
                    else:
                        return flipped_img
        if log: y_new = self.model.predict(flipped_img.reshape([1]+list(img.shape)), verbose = 0)[label.astype(bool).reshape(1,len(label))][0]
        if log:
            return flipped_img, y, y_new, y_log, r_log, img_log
        else:
            return flipped_img