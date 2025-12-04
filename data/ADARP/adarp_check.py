import os
import numpy as np
import csv
from argparse import ArgumentParser
import neurokit2 as nk
import math

participants_folder_names = [
    'Part 101C',
    'Part 102C',
    'Part 104C',
    # 'Part 105C',
    # 'Part 106C',
    # 'Part 107C',
    # 'Part 108C',
    # 'Part 109C',
    # 'Part 110C',
    'Part 111C',
    # 'Part 112C'
]

# 40 minutes, 20 minutes before the event and 20 minutes after the event
tag_segment_length_seconds = 40 * 60
# not-stress data is extracted 60 minutes before and after the event markers.
not_stress_buffer_from_tag = 60 * 60

window_length_seconds = 30
window_step = 15  # 50% overlap
overlap_percent = 0.5

eda_sample_rate = 4
ppg_sample_rate = 64


def get_tag_timestamps(tag_file):
    """
        Open the tag files and retun the tag timestamps as an array.

    :param tag_file: Path to the tags file.
    """

    tag_timestamps = []

    # count = 0
    # for line in open(tag_file): count += 1

    # if count < 2:
    #     return tag_timestamps

    # print(f"{count - 1} tags in {tag_file}")
    with open(tag_file, "r") as read_file:
        csv_reader = csv.reader(read_file)
        # skip the header line
        # next(csv_reader)
        for row in csv_reader:
            # print(row)
            unix_time = float(row[0])
            tag_timestamps.append(unix_time)

    return tag_timestamps


def get_sensor_data(file_path):
    """
    Load data from a text file located at file_path.
    :param file_path: path to the text file

    """
    data = []
    try:
        data = np.genfromtxt(file_path, delimiter=',')
    except:
        print("Error reading the file {}".format(file_path))

    return data


def get_segments_between_timestamps(data_array, tag_timestamps, pre_and_post_event_marker_len=60 * 60, segments=[],
                                    participant_id=None):
    """
        Extract sensor segment for the not-stress class between event markers. For a given event marker
        timestamp we extract sensor segment until one hour before the event marker and one hour after the event
        marker.

        Param
        ================================
        data_array -- sensor data array
        tag_timestamps -- timestamps of tags to extract data around of
        pre_and_post_event_marker_len -- Time duration to skip data points pre and post event marker
        segments -- Array to store the extracted segments
    """

    if (len(data_array) == 0):
        return segments

    if len(tag_timestamps) == 0:
        segments.append(data_array[2:])
        return segments
    else:
        # extract start time, sampling freq, and n_observations
        start_time = data_array[0]
        sampling_freq = data_array[1]
        try:
            if len(start_time):
                start_time = start_time[0]
        except:
            start_time = start_time

        try:
            if len(sampling_freq):
                sampling_freq = sampling_freq[0]
        except:
            sampling_freq = sampling_freq

        # number of samples to skip before and after the event
        n_observation = int(pre_and_post_event_marker_len * sampling_freq)

        # create the tags, add the start and end time into tags
        tags = [start_time]
        tags.extend(tag_timestamps)
        tags.append(tags[0] + len(data_array) / sampling_freq)

        # sensor data and the length
        data = data_array[2:]
        data_length = len(data)

        # for each tag in the tags array
        for i in range(len(tags)):
            j = i + 1
            if j >= len(tags):
                # if at the end, break free
                break

            # get the starting and end point for the sensor segment.
            start_tag = tags[i]  # this is the position of start tag
            end_tag = tags[j]  # this is the position of the end tag

            #             print("Current tags pair ", (start_tag, end_tag))
            # the positions in the array
            here_ = int((
                                start_tag - start_time) * sampling_freq + n_observation)  # pre_and_post_event_marker_len after the event
            there_ = int((
                                 end_tag - start_time) * sampling_freq - n_observation)  # pre_and_post_event_marker_len before the event

            #             print("Indices ", (here_, there_))
            # if there are data points between the start and end points, extract those data points else ignore them
            if ((there_ - here_) > 0):
                pp = data[here_:there_]
                segments.append(pp)

        return segments


# 从带有tag的文件夹中提取非压力？
def not_stressed_data_from_all_files(data_folder, segment_length_to_skip=not_stress_buffer_from_tag,
                                     save_part_data=False, output_folder=None,
                                     segment=False):
    """
        Extract data for not-stressed class from all folders.

        Param
        ===================
        data_folder -- path to the data
        segment_length_to_skip -- length of time to skip before and after a tag event
        save_part_data -- whether to save the participants data or not (default - false)
        output_folder -- path to the directory to save the data (default - none)
        segment -- whether to run sliding window or not

        Return
        ===================
        Sensor segment for EDA, BVP, HR, ACC, and TEMP
        + participant IDs for each segment
    """
    # data containers
    eda_data = []
    scr_data = []
    scl_data = []
    bvp_data = []
    peak_data = []
    participant_ids = []

    # for each participant
    for p in participants_folder_names:
        print(p)
        part_eda_data = []
        part_bvp_data = []
        participants_folder_path = os.path.join(data_folder, p)
        part_subfolders = os.listdir(participants_folder_path)

        # for each sub-folder in the participant's folder
        for sub in part_subfolders:
            print(sub)
            path = os.path.join(participants_folder_path, sub)

            # get the tag events in this folder
            tag_timestamps = get_tag_timestamps(os.path.join(path, "tags.csv"))

            # --- EDA ---
            eda_values = get_sensor_data(os.path.join(path, "EDA.csv"))
            if len(eda_values) != 0:
                part_eda_data = get_segments_between_timestamps(
                    eda_values, tag_timestamps, segment_length_to_skip, part_eda_data, p
                )

            # --- BVP ---
            ppg_values = get_sensor_data(os.path.join(path, "BVP.csv"))  # 修正路径
            if len(ppg_values) != 0:
                part_bvp_data = get_segments_between_timestamps(
                    ppg_values, tag_timestamps, segment_length_to_skip, part_bvp_data, p
                )

        # --- Process EDA ---
        for dt in part_eda_data:
            if len(dt) == 0:
                continue
            eda, eda_info = nk.eda_process(dt, sampling_rate=eda_sample_rate)
            scr = eda["EDA_Phasic"]
            scl = eda["EDA_Tonic"]
            eda_c = eda["EDA_Clean"]
            eda_c = nk.signal_filter(
                eda_c, sampling_rate=eda_sample_rate,
                lowcut=None, highcut=1, method='butterworth', order=4
            )
            eda_data.append(eda_c)
            scr_data.append(scr)
            scl_data.append(scl)
            participant_ids.append(p)

        # --- Process BVP ---
        for dt in part_bvp_data:
            if len(dt) == 0:
                print("Warning:", p, "empty_eda_segment")
                continue
            ppg = nk.ppg_clean(dt, ppg_sample_rate)
            try:
                ppg_peak_index = nk.ppg_findpeaks(ppg, sampling_rate=ppg_sample_rate)["PPG_Peaks"]
                ppg_peak = np.zeros_like(ppg)
                ppg_peak[ppg_peak_index] = 1
                peak_data.append(ppg_peak)
            except IndexError as e:
                print("Warning: No PPG peaks detected for this segment.")
                peak_data.append([])
            bvp_data.append(ppg)
            participant_ids.append(p)

    return list(zip(eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids))


if __name__ == '__main__':
    # data_folder = r"E:\dataset\ADARP\Sensor Data"
    data_folder = r"/home/xjc/data/ADARP/Sensor Data/"
    data_to_save = []

    non_stress_data = not_stressed_data_from_all_files(data_folder)

    for (eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids) in non_stress_data:
        print(len(eda_data))
