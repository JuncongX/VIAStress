import os
import numpy as np
import csv
from argparse import ArgumentParser
import neurokit2 as nk
import math
import re
import json
# from noise_ratio import compute_ppg_noise_ratio, compute_eda_noise_ratio

# from data.CAN_STRESS.noise_ratio import compute_ppg_noise_ratio, compute_eda_noise_ratio

# 60 seconds, 30 seconds before the event and 30 seconds after the event
tag_segment_length_seconds = 60

window_length_seconds = 30
window_step = 10

base_dir = "/home/xjc/data/CAN_Stress/"

save_npy_name = "CAN_STRESS_clip{0}s_multi.npy".format(window_length_seconds)

json_path = "logbook_parsed_by_session.json"

eda_sample_rate = 4
ppg_sample_rate = 64


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


def get_loglook_to_timestamps(loglook_root):
    loglook_path = os.path.join(loglook_root, "logbook.xlsx")


def extract_segments_around_tags(data, tags, segment_size):
    """
        labeled as stress data
        Given data array, tags array and window size extract window size segments
        from the data array around the tags.

    :param data: Data array
    :param tags: An array with tag event times
    :param segment_size: Segment size in seconds

    """
    # return array
    segments = []

    # get the start time: expressed as unit timestamp in UTC i.e., seconds from Jan 1 1970
    start_time = data[0]
    print(start_time)

    # get the sampling frequency expressed in Hz
    sampling_freq = data[1]

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

    # get the sensor data and data length
    sensor_data = data[2:]
    data_length = len(sensor_data)

    # the timestamp corresponding to the last data value
    end_time = start_time + (data_length / sampling_freq)

    # the number of data samples before and after the timestamps
    n_obs = int((segment_size // 2) * sampling_freq)

    skipped_tags = 0

    # for each time stamp in tags
    for timestamp in tags:
        # if the timestamp is within the sensor time array
        if (timestamp >= start_time) & (timestamp <= end_time):
            # how far is the timestamp from the start time.
            difference = int(timestamp - start_time)

            # get the index in the sensor data array, based on the difference of tag timestamp
            position = int(difference * sampling_freq)

            # check is there are enough data points to extract 1 hour of data around the tag.
            check_threshold = 30 * 60 * sampling_freq  # 30 minutes before and after the tag
            if ((position - check_threshold) < 0) | ((position + check_threshold) > data_length):
                # print("not enough data for 1 hour segment length")
                skipped_tags += 1
                continue

            # window segment position in the data array
            from_ = position - n_obs
            to_ = position + n_obs

            # here we can insert logic to make sure that we have sensor data of length segment_size
            if (from_ < 0) | (to_ > data_length):
                # skip this segment if length is not equal to the segment_size
                print(f"skipping segment From: {from_}, To: {to_}, Data Len: {data_length}")
                continue
            else:
                # get the data segment
                seg = sensor_data[from_:to_]
                segments.append(seg)

            # necessary to make sure that we dont have invalid indices
            # if (from_ < 0):
            #     from_ = 0
            # if (to_ > data_length):
            #     to_ = data_length

    # if skipped_tags != 0:
    #     print(f"Skipped {skipped_tags} ", end=" ")
    return segments


def get_bvp_data_around_tags(data_folder, tag_timestamps, segment_size):
    """
        Get BVP segments from the BVP CSV file in data_folder with tag_timestamps
        for segment length of segment_size

    :param data_folder: Path to the folder containing the BVP file
    :param tag_timestamps: An array containing the tag event markers.
    :param segment_size: Segment length in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/BVP.csv"
    sensor_data = get_sensor_data(file_path)

    if len(sensor_data) == 0:
        return []
    else:
        return extract_segments_around_tags(sensor_data, tag_timestamps, segment_size)


def get_eda_data_around_tags(data_folder, tag_timestamps, segment_size):
    """
        Get EDA segments from the EDA CSV file in data_folder with tag_timestamps
        for segment length of segment_size

    :param data_folder: Path to the folder containing the EDA file
    :param tag_timestamps: An array containing the tag event markers.
    :param segment_size: Segment size in seconds

    """

    # load the data from EDA.csv
    file_path = data_folder + "/EDA.csv"
    processed_EDA = []

    sensor_data = get_sensor_data(file_path)

    if len(sensor_data) == 0:
        return processed_EDA

    segments = extract_segments_around_tags(sensor_data, tag_timestamps, segment_size)
    for p in segments:
        processed_EDA.append(p)

    return processed_EDA


def extract_data_around_tags(matched_folders, sessions, segment_length=tag_segment_length_seconds):
    """
        Extract sensor segment around tag event markers.

        Param
        ===================
        data_folder -- path to the data
        segment_length -- length of the sensor segment to extract in seconds
        save_part_data -- whether to save the participants data or not (default - false)
        output_folder -- path to the directory to save the data (default - none)

        Return
        ===================
        Sensor segment for EDA, BVP
        + participant IDs for each segment
    """
    # data containers
    eda_data = []
    scr_data = []
    scl_data = []
    bvp_data = []
    peak_data = []
    participant_ids = []
    stress_rates = []

    # for each participant
    for si in sessions:
        mf = matched_folders[si]
        part_eda_data = []
        part_bvp_data = []

        print(mf)

        # for each sub-folder in the participant's folder

        # get the tag events in this folder
        # tag_timestamps = get_tag_timestamps(os.path.join(path, "tags.csv"))
        tag_timestamps = [rec["timestamp"] for rec in logbook[si]]
        print(tag_timestamps)
        stress_rate = [rec["Stress_Rating"] for rec in logbook[si]]
        ids = [rec["ID"] for rec in logbook[si]]

        if len(tag_timestamps):
            eda_values = get_eda_data_around_tags(mf, tag_timestamps, segment_length)
            if len(eda_values):
                part_eda_data.extend(eda_values)

            ppg_values = get_bvp_data_around_tags(mf, tag_timestamps, segment_length)
            if len(ppg_values):
                part_bvp_data.extend(ppg_values)

        # process EDA
        for (eda_s, ppg_s, sr, id) in zip(part_eda_data, part_bvp_data, stress_rate, ids):
            # eda_nr = compute_eda_noise_ratio(eda_s)
            # ppg_nr = compute_ppg_noise_ratio(ppg_s)
            # if eda_nr > 0.3 or ppg_nr > 0.3:
            #     print(eda_nr, ppg_nr)
            #     continue

            try:
                eda, eda_info = nk.eda_process(eda_s, sampling_rate=eda_sample_rate)
                scr = eda["EDA_Phasic"]
                scl = eda["EDA_Tonic"]
                eda_c = eda["EDA_Clean"]
                eda_c = nk.signal_filter(eda_c, sampling_rate=eda_sample_rate, lowcut=None, highcut=1,
                                         method='butterworth', order=4)

                ppg = nk.ppg_clean(ppg_s, ppg_sample_rate)
                ppg_peak_index = nk.ppg_findpeaks(ppg, sampling_rate=ppg_sample_rate)["PPG_Peaks"]
                ppg_peak = np.zeros_like(ppg)
                ppg_peak[ppg_peak_index] = 1
            except Exception as e:
                print(e)
                print("Warning: signal process failed")
                continue
            eda_data.append(eda_c)
            scr_data.append(scr)
            scl_data.append(scl)
            bvp_data.append(ppg)
            peak_data.append(ppg_peak)
            participant_ids.append(id)
            stress_rates.append(sr)
            print("data_len", len(eda_data))
    return list(zip(eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids, stress_rates))


if __name__ == '__main__':
    with open(json_path, "r", encoding="utf-8") as f:
        logbook = json.load(f)

    # 获取所有 E4_Sessions
    sessions = set(str(k) for k in logbook.keys())

    folder_pattern = re.compile(r"^.+__([0-9]{7})$")  # 最后7位是 E4_Sessions

    matched_folders = {}

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        match = folder_pattern.match(folder_name)
        if match:
            session_id = match.group(1)
            if session_id in sessions:
                matched_folders[session_id] = folder_path
    # print(matched_folders)

    data_to_save = []

    datas = extract_data_around_tags(matched_folders, sessions)

    for (eda_data, scr_data, scl_data, bvp_data, peak_data, participant_id, stress_rate) in datas:
        print(participant_id, stress_rate)
        total_times = (len(bvp_data) - (window_length_seconds - window_step) * ppg_sample_rate) / (
                window_step * ppg_sample_rate)
        for times in range(math.floor(total_times)):
            ppg_start_index = int(times * window_step * ppg_sample_rate)
            ppg_end_index = int(ppg_start_index + window_length_seconds * ppg_sample_rate)
            ppg_signal_ = bvp_data[ppg_start_index: ppg_end_index]

            eda_start_index = int(times * window_step * eda_sample_rate)
            eda_end_index = int(eda_start_index + window_length_seconds * eda_sample_rate)
            eda_signal_ = eda_data[eda_start_index: eda_end_index]
            scr_signal_ = scr_data[eda_start_index: eda_end_index]
            scl_signal_ = scl_data[eda_start_index: eda_end_index]
            ppg_peak_ = peak_data[ppg_start_index: ppg_end_index]

            print(ppg_signal_.shape)
            print(scr_signal_.shape)

            datas = [participant_id, stress_rate, ppg_signal_, scl_signal_, scr_signal_, eda_signal_, ppg_peak_]
            data_to_save.append(datas)

    np.save(save_npy_name, np.array(data_to_save, dtype=object))
