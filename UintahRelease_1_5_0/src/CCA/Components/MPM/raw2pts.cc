/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream.h>
#include <fstream.h>

// To compile:  g++ -o raw2pts raw2pts.cc
// usage:  
//raw2pts xsize ysize zsize dx dy dz lx ly lz hx hy hz threshold < Stack.raw"

main(int argc, char *argv[])
{
  unsigned char intensity;
  ofstream vesselfile("vessel.pts");
  ofstream collagenfile("collagen.pts");

  cout << "argc = " << argc << endl;

  // number of pixels in x and y and number of slices in z
  int xsize_raw = atoi(argv[1]);
  int ysize_raw = atoi(argv[2]);
  int zsize_raw = atoi(argv[3]);

  // physical size of each voxel dx, dy, dz
  double dx = atof(argv[4]);
  double dy = atof(argv[5]);
  double dz = atof(argv[6]);

  int lx,ly,lz,hx,hy,hz,threshold;

  if(argc>8){
    // lower bounds of useable image
    lx = atoi(argv[7]);
    ly = atoi(argv[8]);
    lz = atoi(argv[9]);

    // upper bounds of useable image
    hx = atoi(argv[10]);
    hy = atoi(argv[11]);
    hz = atoi(argv[12]);
    threshold=atoi(argv[13]);
  }
  else{
    lx=0;ly=0;lz=0;
    hx=xsize_raw;
    hy=ysize_raw;
    hz=zsize_raw;
    threshold=atoi(argv[7]);
  }

  // Scan stdin and write the intensity to stdout as text 
  for(int k = 0; k<zsize_raw; k++){
   for(int j = 0; j<ysize_raw; j++){
    for(int i = 0; i<xsize_raw; i++){
       intensity = getc(stdin);
       if((j>=ly) && (j<hy) && (i>=lx) && (i<hx) && (k>=lz) && k<hz){ 
         double x = ((double)i - (double)lx)*dx + dx/2.;
         double y = ((double)j - (double)ly)*dy + dy/2.;
         double z = ((double)k - (double)lz)*dz + dz/2.;
         if(intensity>=threshold){ // vessel
           vesselfile << x << " " << y << " " << z << endl;
         }
         if(intensity<threshold){  // collagen
           collagenfile << x << " " << y << " " << z << endl;
         }
       }
    }
   }
  }
}
