import os
import numpy as np
import csv
from argparse import ArgumentParser
import neurokit2 as nk
import math

# local imports
# import utils as utl

# import filters as filters
# import preprocessing as preprocessing

participants_folder_names = [
    'Part 101C',
    'Part 102C',
    'Part 104C',
    'Part 105C',
    'Part 106C',
    'Part 107C',
    'Part 108C',
    'Part 109C',
    'Part 110C',
    'Part 111C',
    'Part 112C'
]

save_npy_name = "ADARP_clip{0}s_multi_101C.npy"

EXCLUDE_LOG = "exclude.log"

def log_excluded(participant_id, reason, additional_info=""):
    """记录被排除的数据"""
    with open(EXCLUDE_LOG, "a") as f:
        f.write(f"{participant_id}\t{reason}\t{additional_info}\n")

window_length_seconds = 30
window_step = 15  # 50% overlap
overlap_percent = 0.5

eda_sample_rate = 4
ppg_sample_rate = 64

# 40 minutes, 20 minutes before the event and 20 minutes after the event
tag_segment_length_seconds = 40 * 60
# not-stress data is extracted 60 minutes before and after the event markers.
not_stress_buffer_from_tag = 60 * 60


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


def extract_segments_around_tags(data, tags, segment_size, participant_id):
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
                log_excluded(participant_id, "segment_too_short", f"Tag {timestamp}")
                continue

            # window segment position in the data array
            from_ = position - n_obs
            to_ = position + n_obs

            # here we can insert logic to make sure that we have sensor data of length segment_size
            if (from_ < 0) | (to_ > data_length):
                # skip this segment if length is not equal to the segment_size
                print(f"skipping segment From: {from_}, To: {to_}, Data Len: {data_length}")
                log_excluded(participant_id, "invalid_index", f"From {from_}, To {to_}, Data Len {data_length}")
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


def get_bvp_data_around_tags(data_folder, tag_timestamps, segment_size, participant_id):
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
        log_excluded(participant_id, "empty_bvp_file", file_path)
        return []
    else:
        return extract_segments_around_tags(sensor_data, tag_timestamps, segment_size, participant_id)


def get_eda_data_around_tags(data_folder, tag_timestamps, segment_size, participant_id):
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
        log_excluded(participant_id, "empty_eda_file", file_path)
        return processed_EDA

    segments = extract_segments_around_tags(sensor_data, tag_timestamps, segment_size, participant_id)
    for p in segments:
        processed_EDA.append(p)

    return processed_EDA


def segment_sensor_reading(values, window_duration, overlap_percentage,
                           sampling_frequency):
    """
        Sliding window segmentation of the values array for the given window
        duration and overlap percentage.

    param values: 1D array of values to be segmented
    param window_duration: Window duration in seconds
    param overlap_percentage: Float value in the range (0 < overlap_percentage < 1)
    param sampling_frequency: Frequency in Hz
    """

    total_length = len(values)
    window_length = sampling_frequency * window_duration
    segments = []
    if (total_length < window_length):
        return segments

    start_index = 0
    end_index = start_index + window_length
    increment_size = int(window_length * (overlap_percentage))

    while (1):
        # print(start_index, end_index)

        # get the segment
        v = values[start_index:end_index]

        # save the segment
        segments.append(v)

        # change the start and end index values
        start_index += increment_size
        end_index += increment_size

        if (start_index > total_length) | (end_index > total_length):
            # print("we are done, no more segments possible")
            break

    segments = np.array(segments).reshape(len(segments), window_length)
    return segments


def get_segments_between_timestamps(data_array, tag_timestamps, pre_and_post_event_marker_len=60 * 60, segments=[], participant_id=None):
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
            else:
                log_excluded(participant_id, "no_data_between_tags", f"Start {start_tag}, End {end_tag}")

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
                log_excluded(p, "empty_eda_segment")
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
                log_excluded(p, "empty_eda_segment")
                continue
            ppg = nk.ppg_clean(dt, ppg_sample_rate)
            try:
                ppg_peak_index = nk.ppg_findpeaks(ppg, sampling_rate=ppg_sample_rate)["PPG_Peaks"]
                ppg_peak = np.zeros_like(ppg)
                ppg_peak[ppg_peak_index] = 1
                peak_data.append(ppg_peak)
            except IndexError as e:
                log_excluded(p, "ppg_process_failed", str(e))
                print("Warning: No PPG peaks detected for this segment.")
                peak_data.append([])
            bvp_data.append(ppg)
            participant_ids.append(p)

    return list(zip(eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids))


# 从没有tag的文件夹中提取非压力 功能被not_stressed_data_from_all_files囊括
def not_stressed_data_from_zero_tags_files(data_folder, save_part_data=False, output_folder=None, segment=False):
    """
        Extract data for not-stressed class from files with zero tag events
        Param
        ===================
        data_folder -- path to the data
        save_part_data -- whether to save the participants data or not (default - false)
        output_folder -- path to the directory to save the data (default - none)
        segment -- whether to run sliding window or not

        Return
        ===================
        Sensor segment for EDA, BVP, HR, ACC, and TEMP
    """
    # data containers
    eda_data = []
    scr_data = []
    scl_data = []
    bvp_data = []
    peak_data = []
    participant_ids = []

    # for each participants
    for p in participants_folder_names:
        part_eda_data = []
        part_bvp_data = []

        # print("Extracting data for participants {}".format(p))
        participants_folder_path = data_folder + p + "/"
        subfolders = os.listdir(participants_folder_path)

        # for each subfolders in the participant folder
        for sub in subfolders:
            path = participants_folder_path + sub
            # get tag timestamps
            tag_timestamps = get_tag_timestamps(path + "/tags.csv")

            if len(tag_timestamps) == 0:
                # load the EDA data
                data = get_sensor_data(path + "/EDA.csv")
                if len(data) != 0:
                    part_eda_data.append(data[2:])

                # BVP Segments
                data = get_sensor_data(path + "/BVP.csv")
                if len(data) != 0:
                    part_bvp_data.append(data[2:])
            # --- Process EDA ---
        for dt in part_eda_data:
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
            ppg = nk.ppg_clean(dt, ppg_sample_rate)
            try:
                ppg_peak_index = nk.ppg_findpeaks(ppg, sampling_rate=ppg_sample_rate)["PPG_Peaks"]
                ppg_peak = np.zeros_like(ppg)
                ppg_peak[ppg_peak_index] = 1
                peak_data.append(ppg_peak)
            except IndexError:
                print("Warning: No PPG peaks detected for this segment.")
                peak_data.append([])
            bvp_data.append(ppg)
            participant_ids.append(p)

    return list(zip(eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids))


# 用于已经验证有tag的文件夹 其实和 extract_data_around_tags 一样，extract_data_around_tags里头自带验证tag的有无
# 这里有难以理解的逻辑 if abs(tag - stamps) < 5: 为什么要判断
# TODO
def extract_segments_for_verified_tags(data_folder, tag_timestamps_folder, segment_length, output_folder=None):
    """
        Extract sensor segment around tag event markers.

        Params
        data_folder -- path to the complete dataset
        tag_timestamps_folder -- path to the folder containing the tag event markers that are verified.
        segment_length -- length of the sensor segment in seconds
        output_folder -- path to store the extracted sensor segment. default None (do not save)

        Return
        Sensor segments for EDA, BVP, ACC, HR, and TEMP
    """

    # total number of segments
    total_segments = 0

    # data containers
    eda_data = []
    hr_data = []
    acc_data = []
    bvp_data = []
    temp_data = []

    for participants_tags_file in os.listdir(tag_timestamps_folder):
        # get the verified tag for the participants
        tag_events = get_tag_timestamps(tag_timestamps_folder + participants_tags_file)
        if (len(tag_events) == 0):
            continue

        # get the participants identifier
        participant_name = participants_tags_file[:9]
        # print(f"{participant_name} has verified tags {tag_events}")

        # the original folder with participant data
        participants_data_folder = data_folder + participant_name + "/"

        # subfolders within the participants data folder.
        subfolders = os.listdir(participants_data_folder)

        # for each verified tag search all the participants subfolders for matching event markers.
        for tag in tag_events:
            # print(f"Searching for verified tag {tag}")

            # for each sub-folder in the participants folder
            for sub in subfolders:
                sub_folder_path = participants_data_folder + sub

                # get the tag events in this folder
                tag_timestamps = get_tag_timestamps(sub_folder_path + '/tags.csv')
                if (len(tag_timestamps) == 0):
                    continue

                # print(f"{participant_name} event markers {tag_timestamps}")
                # if there are tag events, and if any verified tags are within this list
                # extract data around the verified tag event timestamp
                for stamps in tag_timestamps:
                    if abs(tag - stamps) < 5:
                        # print(f"Verified tag {tag}, event marker {stamps}")
                        total_segments += 1

                        values = get_eda_data_around_tags(sub_folder_path, [stamps], segment_length)
                        if len(values):
                            eda_data.extend(values)

                        values = get_bvp_data_around_tags(sub_folder_path, [stamps], segment_length)
                        if len(values):
                            bvp_data.extend(values)

    return np.array(eda_data), np.array(hr_data), np.array(acc_data), np.array(bvp_data), np.array(temp_data)


# 从带有tag的文件夹中提取压力 前60min 后60min
# TODO
def extract_data_around_tags(data_folder, segment_length=tag_segment_length_seconds, save_part_data=False,
                             output_folder=None):
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

            if len(tag_timestamps):
                eda_values = get_eda_data_around_tags(path, tag_timestamps, segment_length, p)
                if len(eda_values):
                    part_eda_data.extend(eda_values)

                ppg_values = get_bvp_data_around_tags(path, tag_timestamps, segment_length, p)
                if len(ppg_values):
                    part_bvp_data.extend(ppg_values)

        # process EDA
        for dt in part_eda_data:
            eda, eda_info = nk.eda_process(dt, sampling_rate=eda_sample_rate)
            scr = eda["EDA_Phasic"]
            scl = eda["EDA_Tonic"]
            eda_c = eda["EDA_Clean"]
            eda_c = nk.signal_filter(eda_c, sampling_rate=eda_sample_rate, lowcut=None, highcut=1, method='butterworth',
                                     order=4)
            eda_data.append(eda_c)
            scr_data.append(scr)
            scl_data.append(scl)
            participant_ids.append(p)

        # process BVP
        for dt in part_bvp_data:
            ppg = nk.ppg_clean(dt, ppg_sample_rate)
            try:
                ppg_peak_index = nk.ppg_findpeaks(ppg, sampling_rate=ppg_sample_rate)["PPG_Peaks"]
                ppg_peak = np.zeros_like(ppg)
                ppg_peak[ppg_peak_index] = 1
                peak_data.append(ppg_peak)
            except IndexError:
                print("Warning: No PPG peaks detected for this segment.")
                peak_data.append([])
            bvp_data.append(ppg)
            participant_ids.append(p)

    return list(zip(eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids))


if __name__ == '__main__':
    # data_folder = r"E:\dataset\ADARP\Sensor Data"
    data_folder = r"/home/xjc/data/ADARP/Sensor Data/"
    data_to_save = []

    stress_data = extract_data_around_tags(data_folder)

    for (eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids) in stress_data:
        print(participant_ids)
        if len(peak_data) == 0:
            continue
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

            datas = [participant_ids, 1, ppg_signal_, scl_signal_, scr_signal_, eda_signal_, ppg_peak_]
            data_to_save.append(datas)

    non_stress_data = not_stressed_data_from_all_files(data_folder)

    for (eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids) in non_stress_data:
        print(participant_ids)
        if len(peak_data) == 0:
            continue
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

            datas = [participant_ids, 0, ppg_signal_, scl_signal_, scr_signal_, eda_signal_, ppg_peak_]
            data_to_save.append(datas)

    # non_stress_data_no_tag = not_stressed_data_from_zero_tags_files(data_folder)
    # for (eda_data, scr_data, scl_data, bvp_data, peak_data, participant_ids) in non_stress_data_no_tag:
    #     print(participant_ids)
    #     if len(peak_data) == 0:
    #         continue
    #     total_times = (len(bvp_data) - (window_length_seconds - window_step) * ppg_sample_rate) / (
    #             window_step * ppg_sample_rate)
    #     for times in range(math.floor(total_times)):
    #         ppg_start_index = int(times * window_step * ppg_sample_rate)
    #         ppg_end_index = int(ppg_start_index + window_length_seconds * ppg_sample_rate)
    #         ppg_signal_ = bvp_data[ppg_start_index: ppg_end_index]
    #
    #         eda_start_index = int(times * window_step * eda_sample_rate)
    #         eda_end_index = int(eda_start_index + window_length_seconds * eda_sample_rate)
    #         eda_signal_ = eda_data[eda_start_index: eda_end_index]
    #         scr_signal_ = scr_data[eda_start_index: eda_end_index]
    #         scl_signal_ = scl_data[eda_start_index: eda_end_index]
    #         ppg_peak_ = peak_data[ppg_start_index: ppg_end_index]
    #
    #         print(ppg_signal_.shape)
    #         print(scr_signal_.shape)
    #
    #         datas = [participant_ids, 0, ppg_signal_, scl_signal_, scr_signal_, eda_signal_, ppg_peak_]
    #         data_to_save.append(datas)

    np.save(save_npy_name.format(window_length_seconds, 3, 0.5, 8),
            np.array(data_to_save, dtype=object))
