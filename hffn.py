import numpy as np
import os
import notebook_util
os.environ['KERAS_BACKEND'] = 'tensorflow'
from util_function import  *
import json            
import warnings, os
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, Input, Activation, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
import pickle
import sys
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from utils import *

# np.random.seed(1337) # for reproducibility
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calc_test_result(result, test_label, test_mask):
    true_label=[]
    predicted_label=[]
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
                true_label.append(np.argmax(test_label[i,j] ))
                predicted_label.append(np.argmax(result[i,j] ))
    
    # print(classification_report(true_label, predicted_label))
    # print("Accuracy {:.4f}".format(accuracy_score(true_label, predicted_label)))
    # print("F1 {:.4f}".format(f1_score(true_label, predicted_label, average='weighted')))
    return ar(true_label), ar(predicted_label)

def segmentation(text, audio, video, size, stride):
        # print('text',text.shape,'audio',audio.shape,'video',video.shape)
        s = stride; length = text.shape[2]
        local = int((length-size)/s) + 1
        if (length-size)%s != 0 :
           k = (length-size)%s
           pad = size - k
           text = np.concatenate((text,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
           audio = np.concatenate((audio,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
           video = np.concatenate((video,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
           local +=1
        input1 =  np.zeros([text.shape[0],text.shape[1],local,3*size])
        fusion = np.zeros([text.shape[0],text.shape[1],local,(size+1)**3])

        for i in range(local):
            text1 = text[:,:,s*i:s*i+size]
            text2 = text1
            text1 = np.concatenate((text1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
            text1 = text1[:,:,:,np.newaxis]

            audio1 = audio[:,:,s*i:s*i+size] 
            audio2 = audio1
            audio1 = np.concatenate((audio1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
            audio1 = audio1[:,:,np.newaxis,:]

            video1 = video[:,:,s*i:s*i+size]  
            video2 = video1
            video1 = np.concatenate((video1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
            video1 = video1[:,:,np.newaxis,:]

            ta = np.matmul(text1,audio1)
  
            ta = np.reshape(ta,[text.shape[0],text.shape[1],(size+1)**2,1])
            tav = np.matmul(ta,video1)
            tav = np.reshape(tav,[text.shape[0],text.shape[1],(size+1)**3])
            fusion[:,:,i,:] = tav
            input1[:,:,i,0:size] = text2
            input1[:,:,i,size:size*2] = video2
            input1[:,:,i,size*2:size*3] = audio2
        return fusion, input1, local


def multimodal(unimodal_activations, args):
    #Fusion (appending) of features
        #[62 63 50] [62 63 150]
    model_save_path = join(args['model_path'], 'hffn')
    model_sub_save_path = join(model_save_path, 'hffn')
    train_mask=unimodal_activations['train_mask'] # 0 or 1
    test_mask=unimodal_activations['test_mask']
    train_label=unimodal_activations['train_label']
    test_label=unimodal_activations['test_label']
    #  concat = Lambda(lambda x: K.concatenate([x[0],x[1]],axis=-1))
    # padd = np.ones([62,63,1])

    text = unimodal_activations['text_train']
    audio = unimodal_activations['audio_train']
    video = unimodal_activations['video_train']
    if args['zero_video']:
        video = np.zeros_like(video)
    fusion, _, _ = segmentation(text, audio, video, args['segmentation_size'], args['segmentation_stride'])

    text = unimodal_activations['text_test']
    audio = unimodal_activations['audio_test']
    video = unimodal_activations['video_test']
    if args['zero_video']:
        video = np.zeros_like(video)
    fusion2, _, _ = segmentation(text, audio, video, args['segmentation_size'], args['segmentation_stride'])

    input_data = Input(shape=(fusion.shape[1],fusion.shape[2],fusion.shape[3]))  #???

    lstm3 = TimeDistributed(ABS_LSTM4(units=3, intra_attention=True, inter_attention=True))(input_data)  # or ABS_LSTM5
    lstm3 = TimeDistributed(Activation('tanh'))(lstm3)  #tanh
    lstm3 = TimeDistributed(Dropout(0.6))(lstm3)   #0.6
    fla = TimeDistributed(Flatten())(lstm3)
    uni = TimeDistributed(Dense(50,activation='relu'))(fla)   ####50
    uni = Dropout(0.5)(uni)
    output = TimeDistributed(Dense(args['train_label'].shape[-1], activation='softmax'))(uni) 
    # output = TimeDistributed(Dense(args['train_label'].shape[-1], activation='linear'))(uni) 
    model = Model(input_data, output)
    # if False:
    if exists(model_save_path) and not args['overwrite_models']: # can't use exists b/c load and save weights
        print('Using saved hffn model')
        model.load_weights(model_sub_save_path)
    else:
        print('Training hffn...')
        model.compile(optimizer='RMSprop', loss='cosine_similarity', sample_weight_mode='temporal')
        # model.compile(optimizer='RMSprop', loss='mse', sample_weight_mode='temporal')
        # model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        class_weight_mask = np.argmax(train_label, axis=-1).astype('float32')
        for class_val, weight in args['class_weights'].items():
            class_weight_mask[class_weight_mask==class_val] = weight
        class_weight_mask = class_weight_mask * train_mask

        model.fit(fusion, train_label,
            epochs=1 if args['test'] else 1000,
            steps_per_epoch=20 if args['test'] else None,
            batch_size=10,
            sample_weight=class_weight_mask if args['average_type'] == 'macro' else train_mask,
            shuffle=True, 
            callbacks=[early_stopping],
            validation_split=0.2,
            verbose=args['verbose'],
        )
        model.save_weights(model_sub_save_path)
    
    result = model.predict(fusion2)
    return calc_test_result(result, test_label, test_mask)


if __name__ == "__main__":
    '''
    Usage
    python3 mosi_acl.py --data mosei --classes 2 --idx 13
    '''
    from utils import *
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimodal", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--fusion", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_raw", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--data", type=str, default='mosei')
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--segmentation_size", type=int, default=2)
    parser.add_argument("--segmentation_stride", type=int, default=2)
    parser.add_argument("--idx", type=str, default='') # saves to different files
    parser.add_argument("--base_path", type=str, default='/z/abwilf/hffn/') # saves to different files
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    from notebook_util import setup_gpu
    setup_gpu(args['gpu'])
    from hffn_imports import *

    base_path = args['base_path']
    dataset_name = args['data']
    num_classes = args['train_label'].shape[-1]
    idx = args['idx']

    if not args['test']:
        init_except_hook(os.path.basename(__file__))
    else:
        print('This is a TEST\n')

    context_out_path = join(base_path, f'{dataset_name}_{num_classes}way_{idx}.pk')
    u = load_pk(context_out_path)

    u = {k:np.array(v) for k,v in list(u.items())}
    multimodal(u, args)

    if not args['test']:
        t.send('Finished running HFFN!')

    print('\n\nEnd time: ', datetime.now().strftime('%I:%M %p %A'))