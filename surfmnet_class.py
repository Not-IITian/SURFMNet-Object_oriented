#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 21:16:02 2020

@author: abhishek
"""
import os
import sys
import tensorflow as tf

#using global functions inside a class..you can just use them as such.
class surfmnet:
    def __init__(self, SURFMNetConfig):
        self.num_evecs = SURFMNetConfig["num_evecs"]
        self.pc_s = tf.placeholder(tf.float32,shape=(None, None, 3))
        self.pc_t = tf.placeholder(tf.float32,shape=(None, None, 3))
        self.source_evecs = tf.placeholder(tf.float32, shape=(None, None, self.num_evecs))
        self.source_evecs_trans = tf.placeholder(tf.float32,shape=(None, self.num_evecs, None))
        self.source_evals = tf.placeholder(tf.float32,shape=(None, self.num_evecs))
        self.target_evecs = tf.placeholder(tf.float32,shape=(None, None, self.num_evecs))
        self.target_evecs_trans = tf.placeholder(tf.float32,shape=(None, self.num_evecs, None))
        self.target_evals = tf.placeholder(tf.float32,shape=(None, self.num_evecs))
        # train\test switch flag
        self.phase = tf.placeholder("bool", None) 
        self.pc_size = SURFMNetConfig["pc_size"]
        self.dims = SURFMNetConfig["dims"]
        self.num_fclayers = SURFMNetConfig['num_fclayers']
        batch = tf.Variable(0)
        #self.bn_decay = get_bn_decay(batch)
        with tf.variable_scope('siamese') as scope:
            o1 = self.model(self.pc_s)
            scope.reuse_variables()
            o2 = self.model(self.pc_t)
                  
        
        
        self.C_est_AB = self.solve_ls(o1,o2)
        self.C_est_BA = self.solve_ls(o2,o1)
        self.cost = self.func_map_layer()
        
    
    def solve_ls(self,A, B):    
    # Transpose input matrices
        At = tf.transpose(A, [0, 2, 1])
        Bt = tf.transpose(B, [0, 2, 1])

    # Solve C via least-squares
        Ct_est = tf.matrix_solve_ls(At, Bt)
    #Ct_est = tf.matrix_solve_ls(At, Bt, l2_regularizer = 0.000001)
        C_est = tf.transpose(Ct_est, [0, 2, 1], name='C_est')
   
        return C_est

    def res_layer(self,x_in, dims_out, scope, phase):
        with tf.variable_scope(scope):
            x = tf.contrib.layers.fully_connected(x_in, dims_out,activation_fn=None,scope='dense_1')
            x = tf.contrib.layers.batch_norm( x,center=True, scale=True,is_training=phase,
                                                variables_collections=["batch_norm_non_trainable_variables_collection"],
                                                scope='bn_1')
            x = tf.nn.relu(x, 'relu')
            x = tf.contrib.layers.fully_connected(x,dims_out,activation_fn=None,scope='dense_2')
            x = tf.contrib.layers.batch_norm( x,center=True,scale=True,is_training=phase,
                                                variables_collections=["batch_norm_non_trainable_variables_collection"],
                                                scope='bn_2')
            
            # If dims_out change, modify input via linear projection
            # (as suggested in resNet)
            if not x_in.get_shape().as_list()[-1] == dims_out:
                x_in = tf.contrib.layers.fully_connected( x_in, dims_out, activation_fn=None,scope='projection')
            x += x_in
            return tf.nn.relu(x)
  
    def model(self,pc_1):
    #source_evecs, source_evecs_trans, source_evals, target_evecs, target_evecs_trans, target_evals, bn_decay=None):
        dims=self.dims
        """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """        
        phase= self.phase
        
        net = {}    
        for i_layer in range(self.num_fclayers):
            with tf.variable_scope("layer_%d" % i_layer) as scope:
                if i_layer == 0:
                    net['fclayer_%d' % i_layer] = self.res_layer(pc_1, dims_out = 128, scope= scope,phase= phase)
                else:
                    net['fclayer_%d' % i_layer] = self.res_layer(net['fclayer_%d' % (i_layer-1)], dims_out = int(dims[i_layer]),scope= scope, phase= phase)
                                           
        #  Project output features on the shape Laplacian eigen functions
        
        layer_C_est = i_layer + 1   # Grab current layer index
        F = net['fclayer_%d' % (layer_C_est-1)]
        A = tf.matmul(self.source_evecs_trans, F) # why trans
        #net['A'] = A
        return A
                    
    #use a parameter with self. and use a memeber function also with self.
    def penalty_bijectivity(self): 
        x_1 = tf.nn.l2_loss(tf.subtract(tf.matmul(self.C_est_AB, self.C_est_BA),tf.eye(tf.shape(self.C_est_AB)[1])))
        x_2 = tf.nn.l2_loss(tf.subtract(tf.matmul(self.C_est_BA, self.C_est_AB),tf.eye(tf.shape(self.C_est_BA)[1])))
   
        return (x_1+x_2)

    def penalty_ortho(self):   
        x_1 = tf.nn.l2_loss(tf.subtract(tf.matmul(tf.transpose(self.C_est_AB, perm=[0, 2, 1]),self.C_est_AB),tf.eye(tf.shape(self.C_est_AB)[1])))
        x_2 = tf.nn.l2_loss(tf.subtract(tf.matmul(tf.transpose(self.C_est_BA, perm=[0, 2, 1]),self.C_est_BA),tf.eye(tf.shape(self.C_est_BA)[1])))
        return (x_1 +x_2)

    def penalty_laplacian_commutativity(self):    
    # Quicker and less memory than taking diagonal matrix
        eig1 = tf.einsum('abc,ac->abc', self.C_est_AB, self.source_evals)
        eig2 = tf.einsum('ab,abc->abc', self.target_evals, self.C_est_AB)
        eig3 = tf.einsum('abc,ac->abc', self.C_est_BA, self.source_evals)
        eig4 = tf.einsum('ab,abc->abc', self.target_evals, self.C_est_BA)
        return tf.nn.l2_loss(tf.subtract(eig2, eig1)) + tf.nn.l2_loss(tf.subtract(eig4, eig3))

    def func_map_layer(self):
         alpha = 10**3  # Bijectivity
         beta = 10**3   # Orthogonality
         gamma = 1      # Laplacian commutativity
    
         E1 = self.penalty_bijectivity()/2

         E2 = self.penalty_ortho()/2

         E3 = self.penalty_laplacian_commutativity()/2
                                                                                 
         loss = tf.reduce_mean(alpha * E1 + beta * E2 + gamma * E3 )
    
     #check this..
         loss /= tf.to_float(tf.shape(self.C_est_AB)[1] * tf.shape(self.C_est_AB)[0])   

         return loss
              
