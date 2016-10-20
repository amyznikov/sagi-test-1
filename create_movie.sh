#! /bin/bash

input_directory="${1}"

if [[ "${input_directory}" == "" ]] ; then
	echo "No input directory specified" 1>&2;
	echo "Usage:" 1>&2;
	echo " ./create_movie.sh <path-to-directory-with-frames>" 1>&2;
	exit 1;
fi	

ffmpeg -framerate 10 -i "${input_directory}/frame%03d.tif" -c:v libx264 -pix_fmt yuv420p -crf 22 -profile:v high -f mp4 movie.mp4 