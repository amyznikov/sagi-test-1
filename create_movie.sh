#! /bin/bash

input_directory="${1}"
filemask="${2}"

if [[ "${input_directory}" == "" ]] ; then
	echo "No input directory specified" 1>&2;
	echo "Usage:" 1>&2;
	echo " ./create_movie.sh <path-to-directory-with-frames>" 1>&2;
	exit 1;
fi	


echo "input_directory='${input_directory}'"
echo "filemask='${filemask}'"

if [[ "${filemask}" == "" ]] ; then
	filemask='frame%03d.tif';
fi

ffmpeg -y -framerate 10 -i "${input_directory}/${filemask}" -c:v libx264 -pix_fmt yuv420p -profile:v high -crf 22  -f mp4 movie.mp4
 