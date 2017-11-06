
# coding: utf-8

# In[1]:

from utils import * 
import numpy as np

URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
f_name = "lastfm-dataset-1K.tar.gz"
dir_name = "lastfm-dataset-1K"
dataset_f_name = "userid-timestamp-artid-artname-traid-traname.tsv"


download_dataset(URL, f_name)
user_data = load_dataset(dir_name, dataset_f_name, None)


# ## Cleaning infrequent data
# First remove infrequent tracks (<10 plays)

# Removing infrequent tracks.
if user_data['track-id'].isnull().sum() > 0:
    user_data = user_data.dropna(axis = 0, subset = ['track-id'])

total_plays = user_data.groupby(["track-id"]).size().reset_index()
total_plays.rename(columns = {0: 'total'}, inplace = True)

# In[8]:

frequent_plays = total_plays[total_plays['total'] >= 10]['track-id']

# In[9]:

# Drop infrequent tracks.
data = user_data[user_data['track-id'].isin(frequent_plays)]

# In[10]:

# Removing infrequent tracks.
if data['userid'].isnull().sum() > 0:
    data = data.dropna(axis = 0, subset = ['userid'])

# In[12]:

user_plays = data.groupby(["userid"]).size().reset_index()
user_plays.rename(columns = {0: 'total'}, inplace = True)


# In[13]:

frequent_users = user_plays[user_plays['total'] >= 10]['userid']

# In[14]:

# Drop infrequent tracks.
data = data[data['userid'].isin(frequent_users)]

# SAVE AFTER DROPPING INFREQUENT DATA
data.to_csv('final-data.csv')    
print('Wrote cleaned data to file..')

# ## Listening Sessions

groups=data.groupby('userid')

import datetime, dateutil.parser
sessions = []
sessions_user = []
for user, indices in groups.groups.iteritems():
    print("------------------------------------------ User ",user)
    user_record = data.loc[indices, :].sort_values(['timestamp'])
    print user_record.shape
#     print user_record.index
    session = [user_record.iloc[0].name]
    for i in range(0, user_record.shape[0] - 1):
        d1 = int(dateutil.parser.parse(user_record.iloc[i]['timestamp']).strftime('%s'))
        d2 = int(dateutil.parser.parse(user_record.iloc[i+1]['timestamp']).strftime('%s'))
        if(d2-d1 < 800 ):
            session.append(user_record.iloc[i+1].name)
        else:
            if(len(session)>10):
                sessions.append(session)
                sessions_user.append(user)
            session=[]
            session.append(user_record.iloc[i+1].name)

# SAVE SESSIONS
with open('listening-sessions.csv','w') as f:
    for s in sessions:
        f.write(", ".join(str(s)))
        f.write("\n")
print('Wrote listening sessions to file..')
            
            
# SAVE ALL MAPPINGS
artists = data[["artist-id", "artist-name"]].drop_duplicates()
artists.to_csv('artists.csv')    
print('Wrote artist mappings to file..')

track = data[["track-id", "track-name"]].drop_duplicates()
track.to_csv('tracks.csv')    
print('Wrote track mappings to file..')

artist_track = data[["track-id", "artist-id"]].drop_duplicates()
artist_track.to_csv('artists-tracks.csv')    
print('Wrote artist-track mappings to file..')



# ## Playcount matrix

playcount = data.groupby(["userid", "track-id"]).size().reset_index()
playcount.rename(columns = {0: 'playcount'}, inplace = True)

from scipy.sparse import csr_matrix
from scipy import io
playcounts = playcount.pivot(index = 'userid', columns = 'track-id', values = 'playcount').fillna(0)
plays_matrix = csr_matrix(playcounts.values)

# SAVE CSR MATRIX
io.mmwrite("playcounts.mtx", plays_matrix)
print('Wrote playcounts matrix to file..')



# List of number of songs ever played by each user. Reduces computation
user_plays = plays_matrix.sum(axis=1) # Sum all columns
# Compute Absolute Rating
absolute_preferences = np.log(plays_matrix / user_plays + 1)
# 2. List of maximum ratings of each item
max_ratings = absolute_preferences.max(axis=0) # Along columns

# Relative Preference: RP(u, i) = AP(u,i) / Max_{c \in U} AP(c, i)
relative_preferences = absolute_preferences / (max_ratings)


# Implicit Rating: R(u, i) = Round up(5 * RP (u,i))
implicit_ratings = np.ceil(5 * relative_preferences)

# SAVE IMPLICIT RATING
io.mmwrite("ratings.mtx", implicit_ratings)
print('Wrote ratings matrix to file..')

# SAVE TRACK IDS
scoring_ranks = [data.loc[s, ['track-id']].values.flatten() for s in sessions]
with open('musics-sentences.txt','w') as f:
    for s in scoring_ranks:
        f.write(" ".join(s))
        f.write("\n")

print('Wrote song2vec set to file..')
                

