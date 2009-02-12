/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

#include <cstdlib>
#include <cstdio>

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
      envmap_image(t,p)=top_image((int)x,(int)y);
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
      envmap_image(t,p)=bottom_image((int)x,(int)y);
    }
  }
  envmap_image.write_image(argv[5], 1);
}
