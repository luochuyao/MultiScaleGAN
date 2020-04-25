import numpy as np
import os
import random
import data.constants as c
import cv2
from scipy.misc import imsave
from datetime import *
from concurrent.futures import ThreadPoolExecutor,wait

_imread_executor_pool = ThreadPoolExecutor(max_workers=16)


def imread_test_img(path, frames):
    img = np.fromfile(path, dtype=np.uint8).reshape(700, 900)
    img = np.row_stack((img, np.zeros((4, 900))))
    img = np.column_stack((img, np.zeros((704, 12))))
    frames[:] = img


def imread_img(path, frames):
    img = np.fromfile(path, dtype=np.uint8).reshape(700, 900)
    frames[:] = img

def imread_clip_img(path,frames,clip_size,start_index):
    img = np.fromfile(path, dtype=np.uint8).reshape(700, 900)
    img = img[start_index[0]:start_index[0]+clip_size[0],start_index[1]:start_index[1]+clip_size[1]]
    frames[:]=img


def quick_read_clip_frame(path_list,clip_size=None,start_index = None):

    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            raise IOError
    frames = np.empty((len(path_list),clip_size[0], clip_size[1]), dtype=np.uint8)
    if img_num==1:
        imread_clip_img(path_list[0],frames[0],clip_size=clip_size,start_index=start_index)
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(imread_clip_img,
                                               path_list[i],
                                               frames[i, :, :, ],
                                               clip_size,
                                               start_index)
            future_objs.append(obj)
        wait(future_objs)

    return frames

def quick_read_frame(path_list,frame_size = None,rescale_rate = None,im_h = None,im_w = None,mode = 'Train'):

    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            raise IOError

    frames = np.empty((len(path_list),frame_size[0], frame_size[1]), dtype=np.uint8)

    if img_num==1:
        imread_img(path_list[0],frames[0])

    for i in range(img_num):
        future_objs = []
        for i in range(img_num):

            if mode=='Train':
                obj = _imread_executor_pool.submit(imread_img,
                                               path_list[i],
                                               frames[i,:,:,])
            else:
                obj = _imread_executor_pool.submit(imread_img,
                                                       path_list[i],
                                                       frames[i,:,:,])
            future_objs.append(obj)
        wait(future_objs)
    return frames



class SequenceRadarDataIterator(object):
    '''
    The Iterator for radar datset
    '''
    def __init__(self,
                 root_path,
                 mode = 'Train',
                 sample_mode = 'Sequence',
                 in_seq_len=4,
                 out_seq_len=1,
                 in_stride = 1,
                 out_stride=1,
                 is_clip=True,
                 clip_width = 144,
                 clip_height = 144,
                 width=900,
                 height=700
                 ):
        '''

        :param root_path: str
                            the root path of datasets
        :param mode: str
                        Can be "Train" or "Test"
        :param sample_mode: str
                            Can be "Random" or "Sequence"
        :param in_seq_len:  int
                            the length of input
        :param out_seq_len: int
                            the length of output
        :param in_stride:   int
                            the stride in same sample
        :param out_stride:  int
                            the stride of slide windows
        :param width: int
                        the width of figure
        :param height:
                        the hight of figure
        '''
        self.__train_folds = ['14_2500_radar','15_2500_radar','16_2500_radar','17_2500_radar']
        self.__test_folds = ['18_2500_radar']
        self.__root_path = root_path
        self.__mode = mode
        if self.__mode == 'Train':
            self.__folds = self.__train_folds
            self.__df = self.get_train_df()
            self.__sample_mode = 'Random'
        elif self.__mode == 'Valid':
            self.__folds = self.__train_folds
            self.__df, self.__df_names, self.__df_name = self.get_valid_df()
            self.__sample_mode = 'Sequence'
        elif self.__mode == 'Test':
            self.__folds = self.__test_folds
            self.__df, self.__df_names, self.__df_name = self.get_test_df()
            self.__sample_mode = 'Sequence'
        else:
            raise ('Mode Error')


        self.__is_clip = is_clip
        if self.__is_clip:

            self.__clip_width = clip_width
            self.__clip_height = clip_height

        self.__begin = 0
        self.__in_seq_len = in_seq_len
        self.__out_seq_len = out_seq_len
        self.__freq = 6
        self.__in_stride = in_stride
        self.__out_stride = out_stride
        self.__width = width
        self.__height = height

        self.__end = self.__begin+self.__in_seq_len+self.__out_seq_len

    def reset_index(self):
        self.__begin = 0
        self.__end = self.__begin + self.__in_seq_len + self.__out_seq_len

    def judge_continue(self,last_file,current_file):

        last_date = datetime(
            int(last_file[10:22][:4]),
            int(last_file[10:22][4:6]),
            int(last_file[10:22][6:8]),
            int(last_file[10:22][8:10]),
            int(last_file[10:22][10:12]),0,0
        )
        current_date = datetime(
            int(current_file[10:22][:4]),
            int(current_file[10:22][4:6]),
            int(current_file[10:22][6:8]),
            int(current_file[10:22][8:10]),
            int(current_file[10:22][10:12]),0,0
        )

        if (current_date-last_date).seconds == 60*6 and (current_date-last_date).days<1:
            return True
        else:
            return False

    def get_test_df(self):
        process_address = []
        process_address_name = []
        process_name = []

        for test_fold in self.__folds:
            files = os.listdir(self.__root_path+test_fold+'/')
            files.sort()
            current_process = []
            current_name = []
            for i,file in enumerate(files):
                if i==0:
                    process_name.append(files[i][10:22])
                    last_date = files[i]
                    current_process.append(self.__root_path+test_fold+'/'+last_date)
                    current_name.append(files[i][10:22])
                    continue
                current_date = files[i]

                if self.judge_continue(last_date,current_date):
                    current_process.append(self.__root_path+test_fold+'/'+current_date)
                    current_name.append(files[i][10:22])
                else:
                    process_name.append(files[i][10:22])
                    process_address.append(current_process)
                    process_address_name.append(current_name)
                    current_process = []
                    current_name = []
                last_date = current_date
            process_address.append(current_process)
            process_address_name.append(current_name)
        print(self.__root_path+test_fold+'/')
        print(process_name)
        print(len(process_name),len(process_address))
        return process_address,process_address_name,process_name

    @property
    def up_year(self):
        return ['01','02','03','04','05','06']

    @property
    def down_year(self):
        return ['07','08','09','10','11','12']

    def get_valid_df(self):

        last_year_files = os.listdir(os.path.join(self.__root_path, self.__train_folds[-1]))
        last_year_files.sort()
        process_address = []
        process_address_name = []
        process_name = []
        current_process = []
        current_name = []

        files = []
        for file in last_year_files:
            if file[10:22][4:6] in self.down_year:
                files.append(file)
            else:
                pass

        for i,file in enumerate(files):
            if i==0:
                process_name.append(files[i][10:22])
                last_date = files[i]
                current_process.append(self.__root_path+ self.__train_folds[-1]+'/'+last_date)
                current_name.append(files[i][10:22])
                continue
            current_date = files[i]

            if self.judge_continue(last_date,current_date):
                current_process.append(self.__root_path+ self.__train_folds[-1]+'/'+current_date)
                current_name.append(files[i][10:22])
            else:
                process_name.append(files[i][10:22])
                process_address.append(current_process)
                process_address_name.append(current_name)
                current_process = []
                current_name = []
            last_date = current_date
        process_address.append(current_process)
        process_address_name.append(current_name)

        return process_address, process_address_name, process_name

    def get_train_df(self):
        train_addresses = []
        for train_fold in self.__train_folds[:-1]:
            files = os.listdir(os.path.join(self.__root_path,train_fold))
            files.sort()
            for file in files:
                train_addresses.append(os.path.join(self.__root_path,train_fold,file))

        last_year_files = os.listdir(os.path.join(self.__root_path,self.__train_folds[-1]))
        last_year_files.sort()
        for file in last_year_files:
            if file[10:22][4:6] in self.up_year:
                train_addresses.append(
                    os.path.join(self.__root_path, self.__train_folds[-1], file)
                )
            else:
                pass

        return train_addresses

    def check_index(self):

        valid_df_address = self.__df[self.__begin:self.__end]

        begin = valid_df_address[0].split('/')[-1][10:22]
        end = valid_df_address[-1].split('/')[-1][10:22]

        if (int(end)-int(begin)) == self.__in_stride*self.__freq*(self.__in_seq_len+self.__out_seq_len-1):
            # print(valid_df_address)
            return True,valid_df_address
        else:

            return False,None


    def random_index(self):
        self.__begin = random.randint(0,len(self.__df)-self.__in_seq_len-self.__out_seq_len-1)
        self.__end = self.__begin++self.__in_seq_len+self.__out_seq_len


    def update_index(self):
        self.__begin = self.__begin+self.__out_stride
        self.__end = self.__end+self.__out_stride

        if self.__end<len(self.__df):
            return False
        else:
            return True

    def generator_clip_index(self):

        width_start_point = random.randint(0,self.__width-self.__clip_width)
        height_start_point = random.randint(0,self.__height-self.__clip_height)

        return height_start_point,width_start_point

    def load_frames(self,current_folds,is_clip = None):

        batch_size = len(current_folds)
        if is_clip is not None:
            pass
        else:
            is_clip = self.__is_clip
        if self.__mode=='Test':
            frame_dat = np.zeros((batch_size, self.__height, self.__width, self.__in_seq_len + self.__out_seq_len),
                                 dtype=np.uint8)

            for i in range(len(current_folds)):
                current_fold = current_folds[i]

                frame_dat[i, :, :, ] = quick_read_frame(current_fold, frame_size=(self.__height, self.__width),mode='Test')

            return frame_dat

        else:
            if is_clip:
                frame_dat = np.zeros((batch_size, self.__clip_height, self.__clip_width, self.__in_seq_len + self.__out_seq_len),
                                     dtype=np.uint8)
                for i in range(len(current_folds)):
                    current_fold = current_folds[i]
                    height_start_point,width_start_point = self.generator_clip_index()
                    frame_dat[i, :, :, ] = quick_read_clip_frame(current_fold,
                                                                 clip_size=(self.__clip_height, self.__clip_width),
                                                                 start_index=(height_start_point,width_start_point)).transpose((1,2,0))

                return frame_dat
            else:

                frame_dat = np.zeros((batch_size, self.__height, self.__width,self.__in_seq_len+self.__out_seq_len),dtype=np.uint8)

                for i in range(len(current_folds)):

                    current_fold = current_folds[i]

                    frame_dat[i, :, :, ] = quick_read_frame(current_fold, frame_size=(self.__height, self.__width))

                return frame_dat

    def read_clip(self,current_folds):
        self.__clip_width = 144
        self.__clip_width = 144

        height_start_point, width_start_point = self.generator_clip_index()
        frame_dat = quick_read_clip_frame(current_folds,
                              clip_size=(self.__clip_height, self.__clip_width),
                              start_index=(height_start_point, width_start_point))

        return frame_dat

    @property
    def number_sample(self):
        return len(self.__df)

    def train_smaple(self,batch_size = 2):
        frame_dats = np.empty(shape=(
            batch_size,
            24,
            self.__clip_height,
            self.__clip_width
        ))
        process_indexs = random.sample(self.train_indexes,batch_size)

        i = 0
        for process_index in process_indexs:
            current_process = self.__df[process_index]
            while len(current_process)<24:
                process_index = random.randint(0, self.number_sample)
                current_process = self.__df[process_index]
            current_process_index = random.randint(0,len(current_process)-24)
            frame_dats[i] = self.read_clip(current_process[current_process_index:24+current_process_index])
            i = i+1
        return frame_dats

    def validation_smaple(self,batch_size = 2):
        frame_dats = np.empty(shape=(
            batch_size,
            24,
            self.__clip_height,
            self.__clip_width
        ))
        process_indexs = random.sample(self.validation_indexes,batch_size)
        i = 0
        for process_index in process_indexs:
            current_process = self.__df[process_index]
            while len(current_process)<24:
                process_index = random.randint(0, self.number_sample)
                current_process = self.__df[process_index]
            current_process_index = random.randint(0,len(current_process)-24)
            frame_dats[i] = self.read_clip(current_process[current_process_index:24+current_process_index])
            i = i+1
        return frame_dats


    def sample_process(self,num = 0):
        frames = []
        assert num<len(self.__df_name) and num>=0
        for index,current_folds in enumerate(self.__df):
            if index == num:
                frame_dats = quick_read_frame(current_folds, frame_size=(self.__height, self.__width),mode='Test')
                return frame_dats,self.__df_names[index],self.__df_name[index]
            else:
                continue
        return None



    def sample(self,batch_size):
        '''
        :param batch_size: int
                            Batch size
        :return:
        '''

        if self.__mode == 'Test':
            batch_size = 1
            assert self.__end < len(self.__df), 'end index large than the number of data'
            count = 0
            clips = np.empty([batch_size * c.TEST_SEG_BLOCKS_WIDTH * c.TEST_SEG_BLOCKS_HEIGHT,
                              c.TEST_HEIGHT,
                              c.TEST_HEIGHT,
                              (self.__in_seq_len + self.__out_seq_len)])
            current_folds = []
            is_stop = False
            while count < batch_size:
                is_useful, valid_df_address = self.check_index()

                if is_useful:
                    current_folds.append(valid_df_address)
                    count = count + 1
                else:
                    pass
                if self.update_index():
                    is_stop = True
                    break
                else:
                    pass
            if len(current_folds)==0:
                return None,None,is_stop
            else:
                pass
            frame_dat = self.load_frames(current_folds,False)[0]
            seq_length = frame_dat.shape[2]
            for t in range(seq_length):
                for i in range(c.TEST_SEG_BLOCKS_HEIGHT):
                    for j in range(c.TEST_SEG_BLOCKS_WIDTH):
                        frame = frame_dat[:,:,t]
                        frame = np.row_stack((frame, np.zeros((20, 900)))).reshape(720, 900)
                        clips[i * c.TEST_SEG_BLOCKS_WIDTH + j,:,:,t] = frame[ i * c.TEST_HEIGHT:(i + 1) * c.TEST_HEIGHT,j * c.TEST_WIDTH:(j + 1) * c.TEST_WIDTH]
            return clips, valid_df_address[-(self.__out_seq_len):][0].split('/')[-1][10:22], is_stop

            pass
        else:

            if self.__sample_mode=='Sequence':
                assert self.__end<len(self.__df),'end index large than the number of data'
                count = 0
                current_folds = []

                while count<batch_size:
                    is_useful,valid_df_address=self.check_index()
                    if is_useful:
                        current_folds.append(valid_df_address)
                        count = count+1
                    else:
                        pass
                    if self.update_index():
                        if len(current_folds)==0:
                            return None,None,True

                        frame_dat = self.load_frames(current_folds)

                        in_frame_dat = frame_dat[:,:,:,:self.__in_seq_len]
                        out_frame_dat = frame_dat[:,:,:,-(self.__out_seq_len):]

                        return frame_dat,np.array(current_folds)[:,-(self.__out_seq_len):],True

                frame_dat = self.load_frames(current_folds)
                in_frame_dat = frame_dat[:,:,:,:self.__in_seq_len]
                out_frame_dat = frame_dat[:,:,:,-(self.__out_seq_len):]

                return frame_dat,np.array(current_folds)[:,-(self.__out_seq_len):],False

            elif self.__sample_mode == 'Random':

                count = 0
                current_folds = []
                while count<batch_size:
                    self.random_index()
                    is_useful,valid_df_address=self.check_index()
                    if is_useful:
                        current_folds.append(valid_df_address)
                        count = count+1
                    else:
                        pass
                frame_dat = self.load_frames(current_folds)
                in_frame_dat = frame_dat[:, :, :, :self.__in_seq_len]
                out_frame_dat = frame_dat[:, :, :, -(self.__out_seq_len):]
                return frame_dat

            else:
                raise ('there is not sample mode')


if __name__ == '__main__':
    pass
