import os
import cv2
import numpy as np
from utils.filter import butter_bandpass_filter, detrend
import math

from utils.POS import Pulse

data_root = r"/home/som/8T/DataSets/ubfc_phys/train_rppg"
save_root = r"/home/som/8T/DataSets/ubfc_phys/pos_rppg_fm"


def get_rppg(X, dir_, task, part):
    pulse = Pulse(35, len(X))
    bvp = pulse.get_pulse(X)
    bvp = zero_mean(bvp)
    bvp = detrend(bvp)
    # bvp = butter_bandpass_filter(bvp, 35, 0.7, 2.5)
    save_path = os.path.join(save_root, r'{0}/T{1}'.format(dir_, task))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if math.isnan(bvp[0]):
        print("nan")
    np.savetxt(os.path.join(save_path, r'rppg_{0}_{1}_T{2}.csv'.format(part, dir_, task)), bvp, fmt='%f',
               delimiter=None)


def skin_segment(bgr_image):
    ycrcb_image = None
    try:
        ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
    except Exception as e:
        print(e)
        print(bgr_image.shape)
    H, W, C = ycrcb_image.shape
    mask = np.zeros((H, W), dtype="uint8")
    cb = ycrcb_image[:, :, 2]
    cr = ycrcb_image[:, :, 1]
    y = ycrcb_image[:, :, 0]
    cb_index = np.logical_and(cb > 77, cb < 127)
    cr_index = np.logical_and(cr > 137, cr < 177)
    y_index = np.logical_and(y > 80, y < 255)
    mask[np.logical_and(cb_index, cr_index, y_index)] = 1
    SKIN_ROI = cv2.add(bgr_image, np.zeros(np.shape(bgr_image), dtype=np.uint8), mask=mask)
    return SKIN_ROI


def zero_mean(signal):
    return signal - np.mean(signal)


if __name__ == '__main__':
    havent_done = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
                   's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
                   's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
                   's44', 's45', 's46', 's47', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
    # havent_done = ['s48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
    # havent_done = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    #                's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
    #                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
    #                's44', 's45', 's46', 's47']

    for dir_ in havent_done:
        person_path = os.path.join(data_root, dir_)
        for task in [1, 2, 3]:
            print(person_path, task)

            face_path = os.path.join(person_path, r"T{0}/resized_face".format(task))
            files_name_face = os.listdir(face_path)
            files_name_face.sort(key=lambda x: int(x.split('.')[0]))

            total_lengh = int(files_name_face[-1].split('.')[0])

            batch = np.zeros((total_lengh, 128, 128, 3))
            try:
                for index in range(total_lengh):
                    frame_face = np.array(cv2.imread(os.path.join(face_path, "{0}.jpg".format(index))))
                    frame_face = cv2.resize(frame_face, (128, 128))
                    frame_face = skin_segment(frame_face)
                    batch[index] = frame_face
            except Exception as e:
                print(e)
                continue
            mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6)
            b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
            mean_rgb = np.stack((r, g, b), axis=1)
            get_rppg(mean_rgb, dir_, task, "all")

    # import matplotlib.pyplot as plt
    #
    # face_path = "E:\\dataset\\ubfc-phys\\3_part\\s11\\T1\\face"
    # files_name_face = os.listdir(face_path)
    # files_name_face.sort(key=lambda x: int(x.split('.')[0]))
    #
    # total_lengh = int(files_name_face[-1].split('.')[0])
    #
    # mean_bgr = []
    # for index in range(total_lengh):
    #     frame_face = np.array(cv2.imread(os.path.join(face_path, "{0}.jpg".format(index))))
    #     # frame_face = cv2.resize(frame_face, (128, 128))
    #     frame_face = skin_segment(frame_face)
    #     frame_mean_bgr = frame_face.sum(axis=(0, 1)) / (frame_face != 0).sum(axis=(0, 1))
    #     mean_bgr.append(frame_mean_bgr)
    # # mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)))
    # mean_bgr = np.array(mean_bgr)
    # b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
    # # b, g, r = zero_mean(b), zero_mean(g), zero_mean(r)
    # # b, g, r = detrend(b), detrend(g), detrend(r)
    # # b, g, r = butter_bandpass_filter(b, 35), butter_bandpass_filter(g, 35), butter_bandpass_filter(r, 35)
    # mean_rgb = np.stack((r, g, b), axis=1)
    #
    # pulse = Pulse(35, len(mean_rgb))
    # bvp = pulse.get_pulse(mean_rgb)
    # bvp = zero_mean(bvp)
    # bvp = detrend(bvp)
    # bvp = butter_bandpass_filter(bvp, 35)
    # plt.figure()
    # bvp = bvp.tolist()
    # plt.plot(range(len(bvp)), bvp, color='b', linestyle='-')
    # plt.show()
