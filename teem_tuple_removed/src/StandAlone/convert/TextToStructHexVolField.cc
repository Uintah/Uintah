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
 *  TextToStructHexVolField.cc
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
// point, one per line, entries separated by white space.  If file doesn't
// have a one line header specifying ni nj nk (white-space separated), the
// user must specify a -noHeader command line argument, followed by ni, nj,
// and nk.
// The SCIRun output file is written in ASCII, unless you specify 
// -binOutput.

#include <Core/Datatypes/StructHexVolField.h>
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

bool header;
bool binOutput;
bool debugOn;
int ni;
int nj;
int nk;

void setDefaults() {
  header=true;
  binOutput=false;
  debugOn=false;
  ni=0;
  nj=0;
  nk=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
      currArg++;
      ni=atoi(argv[currArg]);
      currArg++;
      nj=atoi(argv[currArg]);
      currArg++;
      nk=atoi(argv[currArg]);
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
  cerr << "\n Usage: "<<progName<<" pts StructHexVolMesh [-noHeader ni nj nk] [-binOutput] [-debug]\n\n";
  cerr << "\t This program will read in a .pts (specifying the x/y/z \n";
  cerr << "\t coords of each point, one per line, entries separated by \n";
  cerr << "\t white space.  If file doesn't have a one line header \n";
  cerr << "\t specifying ni nj nk (white-space separated), the user must \n";
  cerr << "\t specify a -noHeader command line argument, followed by ni, \n";
  cerr << "\t nj, and nk.  The SCIRun output file is written in ASCII, \n";
  cerr << "\t unless you specify -binOutput.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 9) {
    printUsageInfo(argv[0]);
    return 0;
  }
#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  setDefaults();

  char *ptsName = argv[1];
  char *fieldName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }

  ifstream ptsstream(ptsName);
  if (header) ptsstream >> ni >> nj >> nk;
  cerr << "number of points = ("<<ni<<" "<<nj<<" "<<nk<<")\n";
  StructHexVolMesh *hvm = new StructHexVolMesh(ni, nj, nk);
  int i, j, k;
  for (i=0; i<ni; i++)
    for (j=0; j<nj; j++)
      for (k=0; k<nk; k++) {
	double x, y, z;
	ptsstream >> x >> y >> z;
	StructHexVolMesh::Node::index_type idx(hvm, i, j, k);
	hvm->set_point(Point(x, y, z), idx);
	if (debugOn) 
	  cerr << "Added point (idx=["<<i<<" "<<j<<" "<<k<<"]) at ("<<x<<", "<<y<<", "<<z<<")\n";
      }
  cerr << "done adding points.\n";

  StructHexVolField<double> *hvf =
    scinew StructHexVolField<double>(hvm, Field::NONE);
  FieldHandle hvH(hvf);
  
  if (binOutput) {
    BinaryPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, hvH);
  } else {
    TextPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, hvH);
  }
  return 0;  
}    
