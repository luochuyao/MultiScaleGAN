# import os
# from scipy.misc import *
# root1 = '/mnt/A/meteorological/experiment_classic_predict_/multi_scale_gan'
# root2 = '/mnt/A/meteorological/experiment_classic_predict_/multi_scale_gan_'
#
# cc = ['c1','c2','c3','c4']
#
# for c in cc:
#     root1_c = os.path.join(root1,c)
#     root2_c = os.path.join(root2,c)
#     processes = os.listdir(root1_c)
#     for process in processes:
#         process_fold1 = os.path.join(root1_c,process)
#         process_fold2 = os.path.join(root2_c,process)
#         files = os.listdir(process_fold1)
#         files.sort()
#         for file in files:
#             img1 = imread(os.path.join(process_fold1, file))
#             img2 = imread(os.path.join(process_fold2, file))
#             for w in range(len(img1)):
#                 for h in range(len(img1[w])):
#                     if img1[w,h]==img2[w,h]:
#                         pass
#                     else:
#                         print(img1[w,h],img2[w,h])



import os
from scipy.misc import *
root1 = '/mnt/A/meteorological/experiment_classic_predict_/multi_scale_wgan'
root2 = '/mnt/A/meteorological/2500_ref_seq/multi_scale_wgan_140000/'

cc = ['c1','c2','c3','c4']
count = 0
for c in cc:
    root1_c = os.path.join(root1,c)
    processes = os.listdir(root1_c)
    for process in processes:
        process_fold1 = os.path.join(root1_c,process)
        files = os.listdir(process_fold1)
        files.sort()
        imgs = []
        for file in files:
            img1 = imread(os.path.join(process_fold1, file))
            imgs.append(img1)
        root_processes = os.listdir(root2)
        for root_process in root_processes:
            root2_process_fold = os.path.join(root2,root_process)
            pred_folds = os.listdir(root2_process_fold)
            for pred_fold in pred_folds:
                root2_imgs = os.path.join(root2_process_fold,pred_fold,'predict')
                files2 = os.listdir(root2_imgs)
                files2.sort()
                # print(files2[0],files[0])
                if files2[0][5:]==files[0]:
                    count = count+1
                    print(count)
                    for i,file2 in enumerate(files2):
                        img2 = imread(os.path.join(root2_imgs,file2))
                        img1 = imgs[i]
                        # img1[img1>80]=0
                        # img1[img1<15]=0
                        # img2[img2>80]=0
                        # img2[img2<15]=0
                        for w in range(len(img1)):
                            for h in range(len(img1[w])):
                                if img1[w,h]==img2[w,h]:
                                    pass
                                else:
                                    print(img1[w,h],img2[w,h])

                else:
                    pass

        # for file in files:
        #     img1 = imread(os.path.join(process_fold1, file))
        #     img2 = imread(os.path.join(process_fold2, file))
        #     for w in range(len(img1)):
        #         for h in range(len(img1[w])):
        #             if img1[w,h]==img2[w,h]:
        #                 pass
        #             else:
        #                 print(img1[w,h],img2[w,h])