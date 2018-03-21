import sqlite3
import json
from datetime import datetime
import time
import os
import numpy as np
import time
import h5py
from helper_functions import *
from utils import *
import scipy
from scipy import ndimage


def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS images(image_name TEXT PRIMARY KEY, x TEXT, y INT)")

# def format_data(data):
#     data = data.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
#     return data
def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []


# def sql_insert_has_parent(commentid,parentid,parent,comment,subreddit,time,score):
#     try:
#         sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
#         transaction_bldr(sql)
#     except Exception as e:
#         print('s0 insertion',str(e))

def sql_insert_data(image_name, x, y):
    try:
        sql = """INSERT INTO images(image_name, x, y) VALUES ("{}","{}",{});""".format(image_name, x, y)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

# def acceptable(data):
#     if len(data.split(' ')) > 1000 or len(data) < 1:
#         return False
#     elif len(data) > 32000:
#         return False
#     elif data == '[deleted]':
#         return False
#     elif data == '[removed]':
#         return False
#     else:
#         return True

# def find_parent(pid):
#     try:
#         sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
#         c.execute(sql)
#         result = c.fetchone()
#         if result != None:
#             return result[0]
#         else: return False
#     except Exception as e:
#         #print(str(e))
#         return False

# def find_existing_score(pid):
#     try:
#         sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
#         c.execute(sql)
#         result = c.fetchone()
#         if result != None:
#             return result[0]
#         else: return False
#     except Exception as e:
#         #print(str(e))
#         return False
    
# if __name__ == '__main__':
    

#     timeframes = ['2015-02', '2015-03', '2015-04', '2015-05']

#     for timeframe in timeframes:
        
#         sql_transaction = []
#         start_row = 0
#         cleanup = 1000000

#         connection = sqlite3.connect('D:/chatbotData/data/{}.db'.format(timeframe))
#         c = connection.cursor()

#         create_table()
#         row_counter = 0
#         paired_rows = 0

#         with open("D:/chatbotData/reddit_data/2015/RC_{}/RC_{}".format(timeframe, timeframe), buffering=1000) as f:
#             for row in f:
#                 row_counter += 1

#                 if row_counter > start_row:
#                     try:
#                         row = json.loads(row)
#                         parent_id = row['parent_id'].split('_')[1]
#                         body = format_data(row['body'])
#                         created_utc = row['created_utc']
#                         score = row['score']
                        
#                         comment_id = row['id']
                        
#                         subreddit = row['subreddit']
#                         parent_data = find_parent(parent_id)
                        
#                         existing_comment_score = find_existing_score(parent_id)
#                         if existing_comment_score:
#                             if score > existing_comment_score:
#                                 if acceptable(body):
#                                     sql_insert_replace_comment(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                                    
#                         else:
#                             if acceptable(body):
#                                 if parent_data:
#                                     if score >= 2:
#                                         sql_insert_has_parent(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
#                                         paired_rows += 1
#                                 else:
#                                     sql_insert_no_parent(comment_id,parent_id,body,subreddit,created_utc,score)
#                     except Exception as e:
#                         print(str(e))
                                
#                 if row_counter % 100000 == 0:
#                     print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))

#                 if row_counter > start_row:
#                     if row_counter % cleanup == 0:
#                         print("Cleanin up!")
#                         sql = "DELETE FROM parent_reply WHERE parent IS NULL"
#                         c.execute(sql)
#                         connection.commit()
#                         c.execute("VACUUM")
#                         connection.commit()

# directory = os.fsencode("D:/NN Data/Normal")
# i = 1
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     print(filename)
#     os.rename("D:/NN Data/normal/" + filename, "D:/NN Data/Normal/" +  "cl" + str(i) + ".jpg")
#     i += 1



if __name__ == "__main__":
    connection = sqlite3.connect("D:/NN Data/data.db")
    c = connection.cursor()
    sql_transaction = []

    create_table()


    directory = os.fsencode("D:/NN Data/Normal")
    for file in os.listdir(directory):
        filename = "D:/NN Data/Normal/" + os.fsdecode(file)
        num_px = 64
        image = scipy.misc.imread(filename)
        image.flatten()
        # sql_insert_data(os.fsdecode(file), image, 0)
        print(np.array2string(image))

