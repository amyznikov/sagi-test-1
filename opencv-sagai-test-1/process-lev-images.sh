#! /bin/bash


for (( i = 2; i <= 22; ++i)); do 

	idx=$(printf "%03d" $i); 
	rm -f outputs-${idx}/* ;
	
	srcdir="/mnt/sdb1/optimaimaging/data/lev/lev, ran (121345)/lev, ran (121345) 0000008/DualAlign/bicubic/ser-${idx}"
	#filemask="blobs.%03d.tif"

				 
	./opencv-sagai-test-1 "${srcdir}" \
		-o "outputs-${idx}" \
		cr=400 \
		a=avgdiv \
		-g \
		gmin=-0.04 \
		gmax=0.04 \
		b=50 \
		f=0 \
		gamma=1 \
		alpha=0.0 || exit 1
	

	../create_movie.sh outputs-${idx} "blobs.%03d.tif"  || exit 1
	mv movie.avi blobs-movie-${idx}.avi
	
	../create_movie.sh outputs-${idx} || exit 1 
	mv movie.avi movie-${idx}.avi
	
done
