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
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   December 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a .pts (specifying the x/y/z coords of
// each point, one per line, entries separated by white space, file
// must have a one line header specifying number of points.  See usage
// string for details.


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

namespace SCIRun {
extern FieldHandle TextPointCloudString_reader(ProgressReporter *pr,
					       const char *file);
}

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;


bool binOutput;
bool debugOn;

void setDefaults() {
  binOutput=false;
  debugOn=false;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg], "-binOutput")) {
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
  cerr << "\n Usage: "<<progName<<" pts PointCloudMesh [-binOutput] [-debug]\n\n";
  cerr << "\t This program will read in a .pts (specifying the x/y/z \n";
  cerr << "\t coords of each point, one per line, entries separated by \n";
  cerr << "\t white space.  The file must contain a one line header \n";
  cerr << "\t specifying number of points.  The strings are to be listed \n";
  cerr << "\t one per line after the points.  The SCIRun output file is \n";
  cerr << "\t written in ASCII unless -binOutput is specified.\n\n";
}

int
main(int argc, char **argv)
{
  if (argc < 3 || argc > 6) {
    printUsageInfo(argv[0]);
    return 0;
  }

#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  setDefaults();

  const char *ptsName = argv[1];
  const char *fieldName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }

  ProgressReporter *pr = scinew ProgressReporter();
  FieldHandle pcH(TextPointCloudString_reader(pr, ptsName));
  
  if (pcH.get_rep() == 0)
  {
    return 0;
  }

  if (binOutput) {
    BinaryPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, pcH);
  } else {
    TextPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, pcH);
  }
  return 0;  
}    
