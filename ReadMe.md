
Sagi task statement:

we aquire 8 retinal images every time. 
we use dual align i2kretina to correct the eye movement.
the 8 frames can be found under raw data aligned.
we take the 8 frames and create a ratio movie - meaning, differntial movie. 
playing the new ratio frames allows us to see the motion and use the motion contrast\calculate velocity for the next steps of the process.
you may see the movie in the Output folder (play with VLC, imageJ, etc).
similar to the first small test, we would like to see your approcah to create a similar movie using the raw data.
if you decide to write a code, please also write your steps in pseudo code; adding the rational behind the step would be helpful.

Thanks and good luck, please feel free to write to amiram@opt-imaging.com and sagier@opt-imaging.com if needed.


======


To create the frame sequence for target movie, the OpenCV C++ library was used.
The C++ source code of application is stored in opencv-sagai-test-1/main.cc.

The  opencv-sagai-test-1/movie.mp4 itself is created from frame sequence using ffmpeg, called from 'create_movie.sh' script.


Brief outline of processing algorithm:
	
	1) Input images are loaded sequentially in the order marked by source image index in the image file name.
	
	2) In order to correct notable differences in brightness and contrast of source images, several steps are involved:
		* For each individual image the 'pseudo flat field' is created using blur kernel of large size 100 pixels;
		
		* The central region of 150 pixels radius is selected from source image and is used to compute the pixels mean 
			average and 'energy power' values. The 'Energy power' is computed as square root from sum of squares of 
			all pixels in this central region;
			
		* The mean average, computed using central region mentioned, is subtracted from ALL pixels of source image.
		  This operation is intended to make the brightness zero-point (bias) nearly equal for all processed images.
		  
		* After the bias correction, the source image is divided by the "pseudo-flat-field" image.
		  This operation is intended to equalize the contrast variations on the individual image,
		  as well as to set the contrast scale of all images nearly equal.
		  
		  Important note is that such 'pseudo-flat-field-correction' completely changes the spatial correlation 
		  between image pixels. But this is not a matter for this task, because of main purpose of this movie is
		  getting acceptable visual quality, not preserving spatial correlation.          

	3) Even after 'pseudo-flat-field-correction' the differences in contrast scale between consecutive images are still 
		notable and leads to unpleasant visual blinks in movie.
	    To correct these blinks, the images were adjusted to single scale, provided by very first image in sequence (first frame), 
	    using the ratio of central region powers as scale factor.
	 
	    Finally, the image difference between consecutive frames is computed as follows: 
	 
	 
	 	`diff_image = current_image * (power_of_first_image / power_of_current_image) - previous_image;`
	 	
	 	    
		   
	4) Final step is the normalization of the 'difference image' histogram, to bring all visually important 
		pixels into comfortable	to eye value range.
		For this purpose the histogram is computed for resulting image, and the range of pixel values is 
		selected to contain most of image energy. 		
		Pixel values are truncated to bounds given from histogram analysis, and then scaled to the range of 0..255 to fit U8 data type.

	5) Scaled and converted to U8 resulting image is saved into output folder, provided with the frame number in the output file name.
	

Brief outline of movie generation algorithm:

	To generate the movie from sequence if images the standard ffmpeg application is used.
	For this example the video codec x264 and mp4 file format is used.
	
	Just to make the video more longer and more interesting to view, the frame sequence was looped several times.
	  
	 
	
	
		           











 


