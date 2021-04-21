from model_module import *


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'DATAGENERATOR'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class datagenerator (tf.keras.utils.Sequence):
#16
    def __init__(self,  inputs_ids, 
                        outputs_ids,
                        inputs_dir,
                        outputs_dir,
                        batch_size = 16,
                        shuffle = False):

        """
        inputs_ids  : 입력할 noisy speech의 데이터 네임 inputs_ids: data name of noisy speech to be input
        outputs_ids : 타겟으로 삼을 clean speech의 데이터 네임 outputs_ids: data name of clean speech to target

        inputs_dir  : 입력할 noisy speech의 파일 경로 inputs_dir: file path of noisy speech to be input
        outputs_dir : 타겟으로 삼을 clean speech의 파일 경로 outputs_dir: file path of clean speech to target
        """
        self.inputs_ids  = inputs_ids
        self.outputs_ids = outputs_ids
        
        self.inputs_dir  = inputs_dir
        self.outputs_dir = outputs_dir
        
        """
        들어올 입력 데이트의 크기만큼 np.arange로 인덱스를 나열 List indices by np.arange as much as the size of the input data to be entered.
        self.batch_size : 배치 사이즈 옵션 self.batch_size: Batch size option
        self.shuffle : 셔플 옵션 self.shuffle: shuffle option
        """
        self.indexes = np.arange(len(self.inputs_ids))
        self.batch_size = batch_size
        
        self.shuffle = shuffle
        self.on_epoch_end()


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        'epoch가 끌날 때마다 인덱스를 다시 shuffle하는 옵션' 'Option to reshuffle indexes every time epoch is turned off'
        self.indexes = np.arange(len(self.inputs_ids))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation__(self, inputs_ids, outputs_ids):
        'Generates data containing batch_size samples'
        '데이터 네임과 데이터 디렉토리로 전체 경로를 잡고 그 경로에서 소리 파일 로드'  'Take the full path to the data name and data directory and load the sound file from that path'
        inputs_path  = os.path.join(self.inputs_dir + inputs_ids)
        outputs_path = os.path.join(self.outputs_dir + outputs_ids)

        # return sampling_rate, sound (smapling_rate = 16000)
        _, inputs  = scipy.io.wavfile.read(inputs_path)
        _, outputs = scipy.io.wavfile.read(outputs_path)
        inputs=inputs/32767
        outputs=outputs/32767
        inputs=inputs.astype(np.float32)
        outputs=outputs.astype(np.float32)
        #print(inputs.datatype)
        return inputs[0:16384], outputs[0:16384]  #change this


    def __len__(self):
        "Denotes the number of batches per epoch"
        '''
        self.id_names : 존재하는 총 이미지 개수를 의미합니다.
        self.batch_size : 배치사이즈를 의미합니다.
        전체 데이터 갯수에서 배치 사이즈로 나눈 것 == 1 epoch당 iteration 수
        '''
        '''
        self.id_names: It means the total number of images that exist.
        self.batch_size: It means the batch size.
        Dividing the total number of data by the batch size == Number of iterations per 1 epoch
        '''
        return int(np.floor(len(self.inputs_ids) / self.batch_size))


    def __getitem__(self, index):
        """
        1. 한 epoch에서 매 배치를 반복할 때마다 해당하는 인덱스를 호출
        2. 각 데이터의 아이디에서 해당 배치의 인덱스를 할당
        3. 리턴할 리스트 정의
        4. 할당한 인덱스의 데이터에서 데이터 제네레이션으로 파일을 x, y를 불러오고 적당한 전처리 (여기선 reshape)
        4. 리턴할 리스트에 .append하여 배치 데이터 셋을 생성
        5. np.array 형태로 바꾸어준 후 리턴
        """
        """
        1. Call the corresponding index for every iteration of the batch in one epoch
        2. Allocate the index of the batch from the ID of each data
        3. Define the list to return
        4. Load x and y files from the allocated index data by data generation and preprocess appropriately (reshape here)
        4. Create batch data set by .append to the list to be returned
        5. Return after changing to np.array format
        """
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        inputs_batch_ids  = [self.inputs_ids[k] for k in indexes]
        outputs_batch_ids = [self.outputs_ids[k] for k in indexes]
        #print("--------------")
        #print(inputs_batch_ids)
        #print(outputs_batch_ids)
        #print("--------------")
        

        inputs_batch_ids  = natsort.natsorted(inputs_batch_ids, reverse = False)
        outputs_batch_ids = natsort.natsorted(outputs_batch_ids, reverse = False)
        
        inputs_list = list()
        output_list = list()

        for inputs, outputs in zip(inputs_batch_ids, outputs_batch_ids):
            
            x, y = self.__data_generation__(inputs, outputs)
            
            x = np.reshape(x, (16384, 1)) 
            y = np.reshape(y, (16384, 1))

            inputs_list.append(x)
            output_list.append(y)

        inputs_list = np.array(inputs_list)
        output_list = np.array(output_list)
        
        return inputs_list, output_list
