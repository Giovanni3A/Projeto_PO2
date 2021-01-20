import googlemaps as gm
import json
import sys
import requests
import numpy as np
import urllib.parse
from itertools import tee
from geopandas import GeoDataFrame

# Example of usage of the googlemaps API
def teste_gm(gmaps, times, n, origins, destinations):
	mode = "driving"
	units = "metric"
	print(len(origins), len(destinations))
	for i in range(n):
		for j in range(n):
			print(i,j,times[i][j],end=": ")
			result = gmaps.distance_matrix(origins=origins[i],destinations=destinations[j], 
				mode=mode, units=units)
			print(result["rows"][0]["elements"][0]["duration"]["value"])

# Makes a travel time request on a partition of the data matrix
def make_req_gm(gmaps, indexes, times,from_,to_):
	mode = "driving"
	units = "metric"
	result = gmaps.distance_matrix(from_, to_, mode=mode, units=units)
	n = len(to_)
	for i in range(len(indexes)):
		for j in range(n):
			times[indexes[i]][j] = result["rows"][i]["elements"][j]["duration"]["value"]


# Recursively partitions the data matrix so the googlemaps API can handle it
def rec(gmaps, indexes, times, from_, to_):
	if len(to_) > 100:
		raise ValueError()
	elif len(from_)*len(to_) >  100:
		n = len(from_)
		rec(gmaps, indexes[:n//2], times, from_[:n//2], to_)
		rec(gmaps, indexes[n//2:], times, from_[n//2:], to_)
	else:
		make_req_gm(gmaps, indexes, times,from_,to_)


# Read a list of lat longs from the input file path and calls the recursive algorithm to
# get the travel times
def get_googlemaps(path):

	gmaps = gm.Client(key=GG_KEY)

	arq = open(path,"r")
	n = int(arq.readline())
	origins = []
	destinations = []
	for line in arq.readlines():
		lat, longi = [float(x) for x in line.split()]
		origins.append((lat,longi))
		destinations.append((lat,longi))
	arq.close()

	times = []
	for i in range(n):
		times.append([0]*n)
	rec(gmaps,range(n),times,origins,destinations)
	return n,times

def gen_distance_matrix(origins, destinations):
	'''
		Use google maps' API to fetch distance matrix between origin and destination (lists of lat-long)
	'''
	if len(origins) !=  len(destinations):
		raise ValueError()
	
	gmaps = gm.Client(key=GG_KEY)

	times = [] #numpy???
	for i in range(n):
		times.append([0]*n)
	rec(gmaps, range(n), times, origins, destinations)

	return n, times