macro "sagai_test_1" {

	var input_directory = "/home/projects/sagi-test-1/input-images";
	var output_directory = "/home/projects/sagi-test-1/output-images";
	var input_images = newArray();

	print("Input Directory: " + input_directory);
	print("Output Directory: " + output_directory);
	
	input_directory_contents = getFileList(input_directory);
	
	
	
	for ( i = 0; i < input_directory_contents.length; ++i ) {
        if ( endsWith( input_directory_contents[i], ".tif") ) {
        	input_images = Array.concat(input_images, input_directory_contents[i]);
        }
	}
	 
	
	Array.sort(input_images);
	print("input_images count = " + input_images.length);

	if ( input_images.length < 2 ) {
		print("At least two images are need to create a movie. Having " + input_images.length + " images");
	}
	else {
	
		prev_image = input_images[0];
		print(prev_image);
		
		
		
	
		for ( i = 1; i < input_images.length; ++i ) {
			next_image = input_images[i];
			print(next_image);
	    }
	}
	
}