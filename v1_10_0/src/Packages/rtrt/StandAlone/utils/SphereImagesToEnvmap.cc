/*
 *  SphereImagesToEnvmap.cc
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   July 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Packages/rtrt/Core/PPMImage.h>
#include <Core/Math/Trig.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::endl;

using namespace rtrt;

int
main(int argc, char **argv) {
  if (argc != 6) {
    cerr << "Usage: "<<argv[0]<<" top_in bottom_in ntheta nphi envmap_out\n";
    exit(0);
  }
  PPMImage top_image(argv[1]);
  PPMImage bottom_image(argv[2]);
  int ntheta = atoi(argv[3]);
  int nphi = atoi(argv[4]);
  PPMImage envmap_image(ntheta, nphi);
  int x_width = top_image.get_width();
  int y_width = top_image.get_width();
  double dtheta = 2*M_PI / ntheta;
  double dphi = M_PI / nphi;
  
  double ph=0;
  int p;
  for (p=0; p<nphi/2; p++, ph+=dphi) {
    double th=0;
    for (int t = 0; t<ntheta; t++, th+=dtheta) {
      // for a particular theta/phi, figure out what x/y it comes from
      double x = -cos(th)*sin(ph)*x_width/2;
      double y = sin(th)*sin(ph)*y_width/2;
      x+=x_width/2;
      y+=y_width/2;
      if (x<0) x=0; else if (x>=x_width) x=x_width-1;
      if (y<0) y=0; else if (y>=y_width) y=y_width-1;
      envmap_image(t,p)=top_image(x,y);
    }
  }

  for (ph=-M_PI/2; p<nphi; p++, ph+=dphi) {
    double th=0;
    for (int t = 0; t<ntheta; t++, th+=dtheta) {
      // for a particular theta/phi, figure out what x/y it comes from
      double x = -cos(th)*sin(ph)*x_width/2;
      double y = sin(th)*sin(ph)*y_width/2;
      x+=x_width/2;
      y+=y_width/2;
      if (x<0) x=0; else if (x>=x_width) x=x_width-1;
      if (y<0) y=0; else if (y>=y_width) y=y_width-1;
      envmap_image(t,p)=bottom_image(x,y);
    }
  }
  envmap_image.write_image(argv[5], 1);
}
