#!/usr/bin/env python3
#-*-encoding: utf-8 -*-

import googlemaps as gm
import json
import sys
import requests
import numpy as np
import urllib.parse
from itertools import tee
from geopandas import GeoDataFrame

from config import GH_KEY, GG_KEY
	
def _debug(times_gg, times_gh):
	if times_gg:
		print("Times Google Maps:")
		for i in range(n):
			for j in range(n):
				print(times_gg[i][j],end=" ")
			print(end="\n")
	if times_gh.any:
		print("Times Graphhopper:")
		n = times_gh.shape[0]
		for i in range(n):
			print(i, times_gh[i])
			print(end="\n")


# the only argument of this script is the path of the file containing a list of lat longs
def main():
	pass
	#n,times_gh = get_graphhopper(sys.argv[1])
	#n,times_gg = get_googlemaps(sys.argv[1])
	#write(n,times_off,"times.txt")


if __name__ == "__main__":
    main()