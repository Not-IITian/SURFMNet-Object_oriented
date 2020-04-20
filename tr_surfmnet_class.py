#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 02:06:44 2019

@author: abhishek
"""
import os
import time
import numpy as np
from scipy.spatial import cKDTree

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import scipy.io as sio
from surfmnet_class import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# Training parameterss
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate.')
flags.DEFINE_integer('batch_size', 4, 'batch size.')
# Architecture parameters
flags.DEFINE_integer('num_fclayers', 4, 'network depth') 
flags.DEFINE_integer('num_evecs', 30, "number of eigenvectors used for representation")
# Data parameters
flags.DEFINE_string('targets_dir', 'Shapes/FAUST_r/MAT_SHOT/','directory with shapes')  
flags.DEFINE_string('files_name', '', 'name common to all the shapes')

#flags.DEFINE_string('targets_dir_te', '../../Unsupervised_FMnet/Shapes/Scape_r_aligned/MAT/','directory with shapes')  

flags.DEFINE_integer('max_train_iter', 12000, '')
flags.DEFINE_integer('num_vertices', 3000, '') 
flags.DEFINE_integer('save_summaries_secs', 500, '') 
flags.DEFINE_integer('save_model_secs', 500, '')
flags.DEFINE_string('log_dir_', 'Training/SCAPE_r_aligned/_30_dm/',
                    'directory to save models and results')
flags.DEFINE_string('matches_dir', 'Matches/SCAPE_r_aligned/30_dm/',
                    'directory to matches')
flags.DEFINE_integer('dim_', 128,'')
flags.DEFINE_integer('decay_step', 200000, help='Decay step for lr decay [default: 200000]')

flags.DEFINE_float('decay_rate', 0.7, help='Decay rate for lr decay [default: 0.7]')

# Globals
dim_=FLAGS.dim_   
flags.DEFINE_list('dims', [dim_,dim_,dim_,dim_, dim_, dim_, dim_], '')      
       
#dim_1_layer = int(FLAGS.dims[0])
#flags.DEFINE_integer('dim_shot', dim_1_layer, '')  
#no_layers = FLAGS.num_fclayers

#last_layer = int(FLAGS.dims[no_layers-1])
#flags.DEFINE_integer('dim_out', last_layer, '') 
	
n_tr = 80
train_subjects,test_subjects = (range(80),range(80,100))
#train_subjects = list(range(n_tr)) + list(range(52,60))
#test_subjects = range(60,70)
main_dir = FLAGS.targets_dir
files_name =FLAGS.files_name
main_dir_te = FLAGS.targets_dir
files_name_te =FLAGS.files_name
 
BATCH_SIZE = FLAGS.batch_size
	
SURFMNetConfig = {
    "pc_size": FLAGS.num_vertices,
    "dim_out": FLAGS.dims[FLAGS.num_fclayers-1],
    "num_fclayers":FLAGS.num_fclayers ,
    "dim_shot": int(FLAGS.dims[0]),
    "dim_" : FLAGS.dim_,
    "BATCH_SIZE" : FLAGS.batch_size,
    "dim_1_layer" :int(FLAGS.dims[0]),
     "num_evecs" :FLAGS.num_evecs,
     "dims" : FLAGS.dims     
}
def get_input_pair(batch_size, num_vertices, dataset):
    
    batch_input = {
        'source_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
        'target_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
        'source_evecs_trans': np.zeros((batch_size,FLAGS.num_evecs,num_vertices)),
        'target_evecs_trans': np.zeros((batch_size,FLAGS.num_evecs,num_vertices)),
        'source_shot': np.zeros((batch_size, num_vertices, 352)),
        'target_shot': np.zeros((batch_size, num_vertices, 352)),
        'source_evals': np.zeros((batch_size, FLAGS.num_evecs)),
        'target_evals': np.zeros((batch_size, FLAGS.num_evecs))
                   }       
    for i_batch in range(batch_size):                      
        i_source = train_subjects[np.random.choice(range(n_tr))]
        i_target = train_subjects[np.random.choice(range(n_tr))]          
        
        batch_input_ = get_pair_from_ram(i_target, i_source, dataset)
        
        batch_input_['source_labels'] = range(np.shape(batch_input_['source_evecs'])[0])
        batch_input_['target_labels'] = range(np.shape(batch_input_['target_evecs'])[0])
        
        joint_lbls = np.intersect1d(batch_input_['source_labels'],batch_input_['target_labels'])
        
        joint_labels_source = np.random.permutation(joint_lbls)[:num_vertices]
        joint_labels_target = np.random.permutation(joint_lbls)[:num_vertices]

        ind_dict_source = {value: ind for ind, value in enumerate(batch_input_['source_labels'])}
        ind_source = [ind_dict_source[x] for x in joint_labels_source]

        ind_dict_target = {value: ind for ind, value in enumerate(batch_input_['target_labels'])}
        ind_target = [ind_dict_target[x] for x in joint_labels_target]
        
        message = "number of indices must be equal"
        assert len(ind_source) == len(ind_target), message
        
        evecs_full = batch_input_['source_evecs']
        #print(evecs_full.shape)
        evecs= evecs_full[ind_source, :]
        evecs_trans = batch_input_['source_evecs_trans'][:, ind_source]
        shot = batch_input_['source_shot'][ind_source, :]
        #print(batch_input_['target_shot'].shape)                
        evals = [item for sublist in batch_input_['source_evals'] for item in sublist] # what?
        batch_input['source_evecs'][i_batch] = evecs
        batch_input['source_evecs_trans'][i_batch] = evecs_trans
        batch_input['source_shot'][i_batch] = shot
        batch_input['source_evals'][i_batch] = evals

        evecs = batch_input_['target_evecs'][ind_target, :]
        evecs_trans = batch_input_['target_evecs_trans'][:, ind_target]
        shot = batch_input_['target_shot'][ind_target, :]
        evals = [item for sublist in batch_input_['target_evals'] for item in sublist]
        batch_input['target_evecs'][i_batch] = evecs
        batch_input['target_evecs_trans'][i_batch] = evecs_trans
        batch_input['target_shot'][i_batch] = shot
        batch_input['target_evals'][i_batch] = evals
    return batch_input


def get_input_pair_test(i_target, i_source, dataset):
    batch_input = {}
    batch_input_ = get_pair_from_ram(i_target, i_source, 'test')
    evecs = batch_input_['source_evecs']
    evecs_trans = batch_input_['source_evecs_trans']
    shot = batch_input_['source_shot']
    evals = [item for sublist in batch_input_['source_evals'] for item in sublist]
    batch_input['source_evecs'] = evecs
    batch_input['source_evecs_trans'] = evecs_trans
    batch_input['source_shot'] = shot
    batch_input['source_evals'] = evals

    evecs = batch_input_['target_evecs']
    evecs_trans = batch_input_['target_evecs_trans']
    shot = batch_input_['target_shot']
    evals = [item for sublist in batch_input_['target_evals'] for item in sublist]
    batch_input['target_evecs'] = evecs
    batch_input['target_evecs_trans'] = evecs_trans
    batch_input['target_shot'] = shot
    batch_input['target_evals'] = evals
    return batch_input

def get_pair_from_ram(i_target, i_source, dataset):   
    input_data = {}        
    if dataset=='train':
        evecs = targets_train[i_source]['target_evecs']
        evecs_trans = targets_train[i_source]['target_evecs_trans']
        shot = targets_train[i_source]['target_shot']
        evals = targets_train[i_source]['target_evals']
        input_data['source_evecs'] = evecs
        input_data['source_evecs_trans'] = evecs_trans
        input_data['source_shot'] = shot
        input_data['source_evals'] = evals
        input_data.update(targets_train[i_target])
       
    elif dataset=='test':
        evecs = targets_test[i_source]['target_evecs']
        evecs_trans = targets_test[i_source]['target_evecs_trans']
        shot = targets_test[i_source]['target_shot']
        evals = targets_test[i_source]['target_evals']
        input_data['source_evecs'] = evecs
        input_data['source_evecs_trans'] = evecs_trans
        input_data['source_shot'] = shot
        input_data['source_evals'] = evals
        input_data.update(targets_test[i_target])   
        
    return input_data

def load_targets_to_ram():
    global targets_train,targets_test
    targets_train,targets_test = ({},{})    
    targets_train = load_subs(train_subjects, main_dir,files_name)
    targets_test= load_subs(test_subjects,main_dir_te, files_name_te)

    
def load_subs(subjects_list, dir_name,f_name): 
    targets = {}
    
    for i_target in subjects_list:             
        target_file = dir_name +f_name +'%.4d.mat' % (i_target)
        #print(target_file)
        #vert_file = v_dir +f_name +'%.4d.mat' % (i_target)   
        input_data = sio.loadmat(target_file)        
        evecs = input_data['target_evecs'][:, 0:FLAGS.num_evecs]
        evecs_trans = input_data['target_evecs_trans'][0:FLAGS.num_evecs,:]
        evals = input_data['target_evals'][0:FLAGS.num_evecs]        
        input_data['target_evecs'] = evecs
        input_data['target_evecs_trans'] = evecs_trans
        input_data['target_evals'] = evals 
        #p_feat = sio.loadmat(vert_file)
        #input_data['target_shot'] =[]        
        #input_data['target_shot'] = p_feat['VERT']             
        targets[i_target] = input_data        
    return targets


      
def run_training():
    
    print('log_dir=%s' % FLAGS.log_dir_)
    if not os.path.isdir(FLAGS.log_dir_):
        os.makedirs(FLAGS.log_dir_)
    if not os.path.isdir(FLAGS.matches_dir):
        os.makedirs(FLAGS.matches_dir)
    print('num_evecs=%d' % FLAGS.num_evecs)

    print('building graph...')
    
    with tf.Graph().as_default():
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
            #optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            #train_op = optimizer.minimize(net_loss,global_step=global_step, aggregation_method=2)
        our_model = surfmnet(SURFMNetConfig)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(our_model.cost,global_step=global_step)		   
        #saver = tf.train.Saver(tf.global_variables())   
                
        print('starting session...')
        iteration = 0
        init = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# Launch the graph
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)    
        #with sv.managed_session(config=config) as sess:
            print('loading data to ram...')
            load_targets_to_ram()
            
            print('starting training loop...')
            while iteration < FLAGS.max_train_iter:               
                iteration += 1
                start_time = time.time()
                input_data = get_input_pair(FLAGS.batch_size, FLAGS.num_vertices, 'train')
                           
                _, cost_val = sess.run([optimizer,our_model.cost],feed_dict={our_model.pc_s: input_data['source_shot'],our_model.pc_t: input_data['target_shot'], \
                                           our_model.source_evecs: input_data['source_evecs'], our_model.source_evecs_trans: input_data['source_evecs_trans'],
                    our_model.source_evals: input_data['source_evals'], our_model.target_evecs: input_data['target_evecs'],
                    our_model.target_evecs_trans: input_data['target_evecs_trans'], our_model.target_evals: input_data['target_evals'], our_model.phase:True})
                         
                duration = time.time() - start_time
                print('train -: loss = %.2f (%.3f sec)'% ( cost_val, duration))
                
                if iteration%2000==0:                    
                    for i_source in range(80,99):     
                        for i_target in range(i_source+1,100):
                            
                            t = time.time()
                            
                            input_data = get_input_pair_test(i_target, i_source, 'test')
                            source_evecs_ = input_data['source_evecs'][:, 0:FLAGS.num_evecs]
                            target_evecs_ = input_data['target_evecs'][:, 0:FLAGS.num_evecs]
        
                            feed_dict = {our_model_te.pc_s: input_data['source_shot'],our_model.pc_t: input_data['target_shot'], \
                                           our_model.source_evecs: input_data['source_evecs'], our_model.source_evecs_trans: input_data['source_evecs_trans'],
                    our_model.source_evals: input_data['source_evals'], our_model.target_evecs: input_data['target_evecs'],
                    our_model.target_evecs_trans: input_data['target_evecs_trans'], our_model.target_evals: input_data['target_evals'], our_model.phase:False}
         
                
                            _, C_est_ = sess.run([optimizer,our_model.C_est_AB], feed_dict=feed_dict)                            
                            Ct = np.squeeze(C_est_).T #Keep transposed
                
                            kdt = cKDTree(np.matmul(source_evecs_, Ct))
                            
                            dist, indices = kdt.query(target_evecs_, n_jobs=-1)
                            indices = indices + 1
                
                            print("Computed correspondences for pair: %s, %s." % (i_source, i_target) +
                                  " Took %f seconds" % (time.time() - t))                
                            params_to_save = {}
                            params_to_save['matches'] = indices
                            #params_to_save['C'] = Ct.T
                            # For Matlab where index start at 1  
                            new_dir = FLAGS.matches_dir + '%.3d-' % iteration + '/'
                            
                            if not os.path.isdir(new_dir):
                                print('matches_dir=%s' % new_dir)        
                                os.makedirs(new_dir)
                            
                            sio.savemat(new_dir +FLAGS.files_name_te + '%.4d-' % i_source + FLAGS.files_name_te + '%.4d.mat' % i_target, params_to_save)
                    
#        
            #saver.save(sess, FLAGS.log_dir_ + '/model.ckpt', global_step=step)
            #writer.flush()
            #sv.request_stop()
            #sv.stop()


def main(_):
    import time
    start_time = time.time()
    run_training()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()

