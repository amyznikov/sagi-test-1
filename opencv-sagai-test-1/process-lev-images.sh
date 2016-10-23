#! /bin/bash


for (( i = 2; i <= 22; ++i)); do 

	idx=$(printf "%03d" $i); 
	rm -f outputs-${idx}/* ;
	
	srcdir="/mnt/sdb1/optimaimaging/data/lev/lev, ran (121345)/lev, ran (121345) 0000008/DualAlign/bicubic/ser-${idx}"
	#filemask="blobs.%03d.tif"

				 
	./opencv-sagai-test-1 "${srcdir}" \
		-o "outputs-${idx}" \
		cr=400 \
		a=sub \
		-g \
		gmin=-0.8 \
		gmax=0.8 \
		b=0 \
		f=0 \
		gamma=1 \
		alpha=0.05 || exit 1
	

	../create_movie.sh outputs-${idx} "blobs.%03d.tif"  || exit 1
	mv movie.mp4 blobs-movie-${idx}.mp4
	
	../create_movie.sh outputs-${idx} || exit 1 
	mv movie.mp4 movie-${idx}.mp4
	
done
