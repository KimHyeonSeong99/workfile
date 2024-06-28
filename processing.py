from scipy.stats import entropy
import os
import pandas as pd
from tqdm import tqdm
window_size = '1s'

def calculate_entropy(column):
    counts = column.value_counts()
    return entropy(counts)

def hex_to_int(hex_str):
    try:
        return int(hex_str, 16)
    except ValueError:
        return 0

def processing(file,path):
    print("start processing file:",file,"check time: ",window_size)
    f = open(path + file+".csv","r")
    global df
    df = pd.read_csv(f)
    tqdm.pandas()

    print(df.head())

    if 'SubClass' not in list(df.keys()):
        print("start append SubClass")
        df['SubClass'] = 'Normal'
        print("end append SubClass")

    print("start calculate prev interver")
    df['Prev_Interver'] = df['Timestamp'].diff()
    df['Prev_Interver'] = df['Prev_Interver'].fillna(0).astype(float)
    print("end calculate prev interver")

    print("start convert ID")
    df['Arbitration_ID'] = df['Arbitration_ID'].progress_apply(lambda x: int(x, 16))
    print("end convert ID")

    print("start calculate ID prev interver")
    df['ID_Prev_Interver'] = df.groupby('Arbitration_ID')['Timestamp'].diff()
    df['ID_Prev_Interver'] = df['ID_Prev_Interver'].fillna(0).astype(float)
    print("end calculate ID prev interver")

    print("start calculate Data prev interver")
    df['Data_Prev_Interver'] = df.groupby(['Arbitration_ID','Data'])['Timestamp'].diff()
    df['Data_Prev_Interver'] = df['ID_Prev_Interver'].fillna(0).astype(float)
    print("end calculate Data prev interver")

    print("start record prev ID")
    df['Prev_ID'] = df['Arbitration_ID'].shift(1)
    df['Prev_ID'] = df['Prev_ID'].fillna(0).astype(int)
    print("end record prev ID")

    print("start frequency processing")
    df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('DateTime', inplace=True)
    df['ID_Frequency'] = df.groupby('Arbitration_ID')['Arbitration_ID'].rolling(window_size).count().reset_index(level=0, drop=True)
    df['Data_Frequency'] = df.groupby(['Arbitration_ID','Data'])['Data'].rolling(window_size).count().reset_index(level=[0,1], drop=True)
    df.reset_index(inplace=True)
    print("end frequency processing")

    # print("start split data")
    # df_split = df['Data'].str.split(expand=True, n=7)
    # df_split.columns = [f'Data_{i}' for i in range(1, 9)]
    # df_split = df_split.fillna('00')
    # df['entropies'] = df_split.progress_apply(calculate_entropy, axis = 1)
    # print("end split data")

    print("start Class Encode")
    df['Class'] = df['Class'].map({'Normal': 0, 'Attack': 1})
    df['SubClass'] = df['SubClass'].map({'Normal': 0, 'Flooding': 1, 'Spoofing': 2, 'Replay': 3, 'Fuzzing': 4})
    print("end Class Encode")

    
    df.drop(['DateTime','Timestamp','Data'],axis = 1, inplace = True, errors = 'ignore')
    print(df.head())
    f.close()
    return  df
   

def main():
    target = ['t','s','f']
    # file_type = input("wirte file type: (T: training file/S: submission file/F: final submission file)")
    for file_type in target:
        file_data = {}
        if ((file_type == 'T') | (file_type == 't')):
            path = os.getcwd() + "/CHCD/Preliminary/Training/"
            filename = "Pre_train_"
            filelist = [filename + "d_0",filename + "d_1",filename + "d_2",filename + "s_0",filename + "s_1",filename + "s_2"]
        elif((file_type == 'S') | (file_type == 's')):
            path = os.getcwd() + "/CHCD/Preliminary/Submission/"
            filename = "Pre_submit_"
            filelist = [filename + "d",filename + "s"]
        elif((file_type == 'F') | (file_type == 'f')):
            path = os.getcwd() + "/CHCD/Final/"
            filename = "Fin_host_session_submit_"
            filelist = [filename + "S"]
        else:
            print("Error: don`t exist type")
            exit()

        source_path = os.getcwd() + "/source/CHCD/"
        for file in filelist:
            processed_file = processing(file,path)
            # writefile = source_path + file + "_proc.csv"
            # processed_file.to_csv(writefile, index=False)
            # print(f"save to {writefile}.")
            file_data[file] = processed_file

        all_data = pd.concat(file_data.values(), ignore_index=True)
        all_data.to_csv(source_path + filename +'total_proc.csv', index=False)


if __name__ == "__main__":
    main()