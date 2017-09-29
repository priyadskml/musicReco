import urllib
import os.path, sys, tarfile
import numpy as np
import pandas as pd

URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
f_name = "lastfm-dataset-1K.tar.gz"
dir_name = "lastfm-dataset-1K"
dataset_f_name = "userid-timestamp-artid-artname-traid-traname.tsv"

# Download dataset if not in directory.
def download_dataset(URL, f_name):
	if not os.path.isfile(f_name):
		print("Downloading dataset...")
		f = urllib.URLopener()
		f.retrieve(URL, f_name)

		print("Extracting dataset...")
		tar = tarfile.open(f_name)	
		tar.extractall()
		tar.close()
		print("Extraction Completed...")
	else:
		print("Dataset Exists...")
	return

# userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name
def load_dataset(dir_name, f_name, n):
	data = pd.read_csv(dir_name + "/" + f_name, nrows=n, delimiter="\t", names=["userid", "timestamp", "artist-id", "artist-name", "track-id", "track-name"], header=None, error_bad_lines=False)
	print("Read File...")
	return data

# def load_metadata(dir_name, f_name):