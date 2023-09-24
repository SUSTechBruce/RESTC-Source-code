# coding=utf-8
import sys
import os
import argparse
import fnmatch
import logging
import moxing as mox
import threading

DS_DIR_NAME = "src"
os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"

LOCAL_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
assert mox.file.exists(LOCAL_DIR)
logging.info("local disk: " + LOCAL_DIR)

parser = argparse.ArgumentParser()

dir_name = os.path.dirname(os.path.abspath(__file__))

###### define args
parser.add_argument("--data_url", type=str, default="", required=True)
parser.add_argument("--train_url", type=str, default="", required=True)
parser.add_argument('--saver_dir', type=str, default="/cache/src/output/", help="the path of saving traing model")

args, _ = parser.parse_known_args()

# copy data to local /cache/lcqmc
logging.info("copying data...")
local_data_dir = os.path.join(LOCAL_DIR, DS_DIR_NAME)
logging.info(mox.file.list_directory(args.data_url, recursive=True))
mox.file.copy_parallel(args.data_url, local_data_dir)


#from nltk import data

#data.path.append(dir_name)
# local_data_dir_data = os.path.join(local_data_dir, "data")
local_data_dir_output = os.path.join(local_data_dir, "output")

if not os.path.exists(local_data_dir_output):
    os.mkdir(local_data_dir_output)

#  data_dir=os.path.join(local_data_dir, "wmt17_en_de") # code/Dual_contrastive/cled_run.py

cmd1 ='''
   pip install entmax
   
   python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 9521 code/Dual_contrastive/train_main_global.py --data_dir {data_dir} --output_dir {save_dir}
  
    '''.format(data_dir=os.path.join(local_data_dir, ""), save_dir=local_data_dir_output)
print(cmd1)
os.system(cmd1)



# copy output data
s3_output_dir = args.train_url
logging.info("copy local data to s3")
logging.info(mox.file.list_directory(local_data_dir_output, recursive=True))
# s3_output_dir=os.path.join(os.path.join(args.data_url, task),args.output_dir)

print('output dir:' + s3_output_dir)
mox.file.copy_parallel(local_data_dir_output, s3_output_dir)