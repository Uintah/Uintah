/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  TextToPointCloudField.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a .pts (specifying the x/y/z coords of each 
// point, one per line, entries separated by white space, file can have 
// an optional one line header specifying number of points... and if it
// doesn't, you have to use the -noPtsCount command-line argument).  
// And the SCIRun output file is written in ASCII, unless you specify 
// -binOutput.

#include <Core/Datatypes/PointCloudField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
#if defined(__APPLE__)
#  include <Core/Datatypes/MacForceLoad.h>
#endif
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool ptsCountHeader;
bool binOutput;
bool debugOn;

void setDefaults() {
  ptsCountHeader=true;
  binOutput=false;
  debugOn=false;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      ptsCountHeader=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-binOutput")) {
      binOutput=true;
      currArg++;
    } else if (!strcmp(argv[currArg], "-debug")) {
      debugOn=true;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" pts PointCloudMesh [-noPtsCount] [-binOutput] [-debug]\n\n";
  cerr << "\t This program will read in a .pts (specifying the x/y/z \n";
  cerr << "\t coords of each point, one per line, entries separated by \n";
  cerr << "\t white space, file can have an optional one line header \n";
  cerr << "\t specifying number of points... and if it doesn't, you \n";
  cerr << "\t have to use the -noPtsCount command-line argument).  And \n";
  cerr << "\t the SCIRun output file is written in ASCII, unless you \n";
  cerr << "\t specify -binOutput.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 6) {
    printUsageInfo(argv[0]);
    return 0;
  }
#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  setDefaults();

  PointCloudMesh *pcm = new PointCloudMesh();
  char *ptsName = argv[1];
  char *fieldName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }

  int npts;
  if (!ptsCountHeader) npts = getNumNonEmptyLines(ptsName);
  ifstream ptsstream(ptsName);
  if (ptsCountHeader) ptsstream >> npts;
  cerr << "number of points = "<< npts <<"\n";
  int i;
  for (i=0; i<npts; i++) {
    double x, y, z;
    ptsstream >> x >> y >> z;
    pcm->add_point(Point(x,y,z));
    if (debugOn) 
      cerr << "Added point #"<<i<<": ("<<x<<", "<<y<<", "<<z<<")\n";
  }
  cerr << "done adding points.\n";

  PointCloudField<double> *pc = 
    scinew PointCloudField<double>(pcm, Field::NONE);
  FieldHandle pcH(pc);
  
  if (binOutput) {
    BinaryPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, pcH);
  } else {
    TextPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, pcH);
  }
  return 0;  
}    
