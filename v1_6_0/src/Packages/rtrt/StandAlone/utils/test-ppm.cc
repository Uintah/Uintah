/*
 *  test-ppm.cc
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
  if (argc != 3) {
    cerr << "Usage: "<<argv[0]<<" ppm_in ppm_out\n";
    exit(0);
  }
  PPMImage image(argv[1]);
  image.write_image(argv[2], 1);
}
