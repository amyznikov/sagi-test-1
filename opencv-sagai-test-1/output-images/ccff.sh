#!/bin/bash


for (( i=1; i < 10; ++i )) {

  for (( j=0; j < 7; ++j )) {
	
    x=$((i*7 + j));	

    src=$( printf "frame%03d.tif" $j);
    dst=$(printf "frame%03d.tif" $x);
    # echo "$src --> $dst";
    cp $src $dst
   
  }  
	
}

    
