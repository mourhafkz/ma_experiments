import numpy as np
import pandas as pd

def convert_ark_to_array(vector_path='xvector.txt'):
    data_path = []
    ivector = []
    f = open(vector_path)
    for line in f.readlines():
        split_data = line.strip().split(' ')
        split_data = [item for item in split_data if item!='']
        name_item = split_data[0]
        features_item = np.array(split_data[2:-1]).astype('float')
        data_path.append({'pic_path': name_item})
        ivector.append(features_item)

    return data_path, ivector

if __name__ == '__main__':
    vector_type = 'i' # or x

    txt_file = f'D:/kaldii/mount_ix_system/{vector_type}vector/{vector_type}vector.txt'
    npz_file = f'D:/kaldii/mount_ix_system/{vector_type}vector/{vector_type}_vector.npz'
    csv_file = f'D:/kaldii/mount_ix_system/{vector_type}vector/{vector_type}vectors.csv'

    # ark/txt to npz --------------------
    data_path, vector = convert_ark_to_array(txt_file)
    np.savez_compressed(npz_file,
                        data_path=data_path,
                        features=vector
                        )
    # npz to csv --------------------
    data = np.load(npz_file, allow_pickle=True)

    filenames_list = [i['pic_path'] for i in data['data_path']]
    vectors = data['features']

    df1 = pd.DataFrame(np.asarray(vectors))
    df2 = pd.DataFrame(filenames_list)
    df_merged = pd.concat([df1, df2], axis=1, ignore_index=True, sort=False)

    df_merged.to_csv(csv_file, index=False, header=False)
    print(df_merged)
    # now open the csv. convert the last column to classes.
