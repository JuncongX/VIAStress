import os

# 定义两个路径
pre_path = "/home/xjc/data/VerBIO_v2/PRE/E4"
post_path = "/home/xjc/data/VerBIO_v2/POST/E4"

# 定义必须存在的四个文件
required_files = {"BVP_RELAX.csv", "EDA_RELAX.csv", "BVP_PPT.csv", "EDA_PPT.csv"}


def valid_subjects(base_path):
    """返回在给定路径下包含所需四个文件的受试者编号集合"""
    subjects = set()
    for subj in os.listdir(base_path):
        subj_path = os.path.join(base_path, subj)
        if os.path.isdir(subj_path):
            files = set(os.listdir(subj_path))
            if required_files.issubset(files):
                subjects.add(subj)
    return subjects


# 获取两个路径下满足条件的受试者
pre_subjects = valid_subjects(pre_path)
post_subjects = valid_subjects(post_path)

# 找出PRE和POST下都满足条件的受试者
common_subjects = sorted(pre_subjects & post_subjects)

# 打印结果
print("PRE和POST下均存在且四个文件齐全的受试者:")
print(common_subjects)
# [
#     'P005', 'P008', 'P023',
#     'P032', 'P035', 'P037', 
#     'P038', 'P041', 'P043',
#     'P044', 'P046', 'P047',
#     'P049', 'P058', 'P062',
#     'P065', 'P071', 'P073'
# ]

common_subjects = sorted(pre_subjects | post_subjects)
print(common_subjects)
# [
#     'P001', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P011', 'P012', 'P013',
#     'P014', 'P016', 'P017', 'P018', 'P020', 'P021', 'P023', 'P026', 'P027', 'P031', 'P032',
#     'P035', 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046',
#     'P047', 'P048', 'P049', 'P050', 'P051', 'P052', 'P053', 'P056', 'P057', 'P058', 'P060',
#     'P061', 'P062', 'P063', 'P064', 'P065', 'P066', 'P067', 'P068', 'P071', 'P072', 'P073'
# ]
