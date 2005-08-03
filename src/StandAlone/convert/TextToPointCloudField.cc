/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/NoData.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
#include <Core/Init/init.h>
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
    return 2;
  }
  SCIRunInit();
  setDefaults();
  typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;
  PCMesh *pcm = new PCMesh();
  char *ptsName = argv[1];
  char *fieldName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 2;
  }

  int npts;
  if (!ptsCountHeader) npts = getNumNonEmptyLines(ptsName);
  ifstream ptsstream(ptsName);
  if (ptsstream.fail()) {
    cerr << "Error -- Could not open file " << ptsName << "\n";
    return 2;
  }
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

  typedef NoDataBasis<double>                DatBasis;
  typedef GenericField<PCMesh, DatBasis, vector<double> > PCField;   


  PCField *pc = scinew PCField(pcm);
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
