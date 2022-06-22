import numpy as np
import glob
import os
import pickle
from tqdm import tqdm
from scipy import spatial
'''
# Baidu ApolloScape data format:
    frame_id, object_id, object_type, position_x, position_y, position_z,object_length, pbject_width, pbject_height, heading
    Read data from $pra_file_path, and split data into clips with $total_frames length. 
            feture: (T, V ,C) 
                    C is the dimension of features, x,y label
                    T is the temporal length of the data. history_frames + future_frames
                    V is the maximum number of objects. zero-padding for less objects. 
'''

# Please change this to your location
data_root = './data/ApolloScape/'

# 3 second * 2 frame/second
history_frames = 6
# 3 second * 2 frame/second
future_frames = 6
total_frames = history_frames + future_frames
frame_step=1
feature_id=[3,4,2,9,6,7]
max_object_nums=115
neighbor_distance = 15

def GenerateData(file_path_list, data_root, is_train=True):
    all_data = []
    # max_object=[]

    for file_path_idx in tqdm(file_path_list):
        # print(file_path_idx)
        with open(file_path_idx, 'r') as reader:
            content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)

        scene_frames = content[:, 0].astype(np.int64)        
        unique_frames = sorted(np.unique(scene_frames).tolist())
        if is_train:
            start_frame_ids = unique_frames[:-total_frames+1]
        else:
            start_frame_ids = unique_frames[::history_frames]
        data_list=[]

        for start_index in start_frame_ids:
            if is_train:
                sample_frames = np.arange(start_index, start_index + total_frames)
            else:
                sample_frames = np.arange(start_index, start_index + history_frames)
            sample_mask = np.any(scene_frames.reshape(-1, 1) == sample_frames.reshape(1, -1), axis=1)
            # sample_object_ids = np.sort(np.unique(content[sample_mask, 1].astype(np.int)))
            sample_object_ids = np.unique(content[sample_mask, 1].astype(np.int))
            # print(start_index,sample_object_ids)
            # le=len(sample_object_ids)
            # max_object.append(le)
            xy_coordinate=content[sample_mask, 3:5].astype(float)
            mean_xy = np.mean(xy_coordinate, axis=0)
            # print('mean_xy',mean_xy)
            
            if is_train:
                neighbor_mask = np.zeros((total_frames, max_object_nums, max_object_nums), dtype=np.bool)
                sample_object_input = np.zeros((total_frames, max_object_nums, len(feature_id)+2), dtype=np.float32)
                sample_object_mask = np.zeros((total_frames, max_object_nums), dtype=np.bool)
            else:
                neighbor_mask = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.bool)
                sample_object_input = np.zeros((history_frames, max_object_nums, len(feature_id)+2), dtype=np.float32)
                sample_object_mask = np.zeros((history_frames, max_object_nums), dtype=np.bool)
                sample_object_origin = np.zeros((history_frames, max_object_nums, 3), dtype=np.int)

            # for every frame
            for frame_idx, frame in enumerate(sample_frames):
                
                exist_object_idx = []
                for object_idx, object_id in enumerate(sample_object_ids):
                    # frame and object
                    matched_obj = content[np.logical_and(content[:, 0] == frame, content[:, 1] == object_id)]
                    if 0 == len(matched_obj):
                        continue
                    obj_feature = matched_obj[0, feature_id]
                    
                    obj_feature[:2]=obj_feature[:2]-mean_xy
                    sample_object_input[frame_idx, object_idx, :-2] = obj_feature

                    # 在时间域内，某个障碍物在某段时间内存在
                    sample_object_mask[frame_idx, object_idx] = True

                    exist_object_idx.append(object_idx)

                    if not is_train:
                        sample_object_origin[frame_idx, object_idx,:3]=matched_obj[0, :3]
                        # print(frame_idx,object_idx,matched_obj[0, :3])
                
                # print(len(exist_object_idx))
                for obj_id_i in exist_object_idx:
                    xy_1 = sample_object_input[frame_idx, obj_id_i, :2]
                    for obj_id_j in exist_object_idx:
                        xy_2 = sample_object_input[frame_idx, obj_id_j, :2]
                        
                        relative_cord = xy_1 - xy_2
                        
                        neighbor_mask[frame_idx, obj_id_i, obj_id_j] = (
                            abs(relative_cord[0]) > neighbor_distance) | (
                                abs(relative_cord[1]) > neighbor_distance)


            # add speed x ,y in dim 4,5
            new_mask = (sample_object_input[1:, :, :2] != 0) * (sample_object_input[:-1, :, :2] != 0).astype(float)      
            sample_object_input[1:, :, -2:] = (
                sample_object_input[1:, :, :2] - sample_object_input[:-1, :, :2]).astype(float) * new_mask
            sample_object_input[0, :, -2:] = 0.            
            
            sample_object_mask = np.expand_dims(sample_object_mask, axis=-1)
            # refine the future masks
            # data['masks'].sum(axis=0) == history_frames表示如果过去帧都在
            #表示在过去帧都存在的情况下对未来的掩码
            if  is_train:
                data = dict(
                    features=sample_object_input, masks=sample_object_mask, mean=mean_xy,
                    neighbors=neighbor_mask)              
                # data['masks'][history_frames-1:] = np.repeat(
                #     np.expand_dims(data['masks'][:history_frames].sum(axis=0) == history_frames, axis=0),
                #     history_frames+1, axis=0) & data['masks'][history_frames-1]
                data['masks'] = data['masks'] & data['masks'][history_frames-1]
            else:
                data = dict(
                    features=sample_object_input, masks=sample_object_mask, mean=mean_xy, 
                    origin=sample_object_origin, neighbors=neighbor_mask)
                # data['masks'][history_frames-1] = np.expand_dims(
                #     data['masks'][:history_frames].sum(axis=0) == history_frames, axis=0) & data['masks'][history_frames-1]                
                data['masks'] = data['masks'] & data['masks'][history_frames-1]

            data_list.append(data)
        
        all_data.extend(data_list)

    all_data = np.array(all_data)  # Train 5010 Test 415
    print(np.shape(all_data))

    # save training_data and trainjing_adjacency into a file.
    if is_train:
        save_path=os.path.join(data_root, 'train_data.pkl')
    else:
        save_path=os.path.join(data_root, 'test_data.pkl')
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data], writer)


if __name__ == '__main__':
    train_file_path_list = sorted(
        glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
    test_file_path_list = sorted(
        glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

    print('Generating Training Data.')
    GenerateData(train_file_path_list,data_root, is_train=True)

    print('Generating Testing Data.')
    GenerateData(test_file_path_list,data_root, is_train=False)
