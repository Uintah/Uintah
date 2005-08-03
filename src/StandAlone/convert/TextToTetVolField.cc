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
 *  TextToTetVolField.cc
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
// doesn't, you have to use the -noPtsCount command-line argument) and a
// .tet file (specifying i/j/k/l indices for each tet, also one per 
// line, again with an optional one line header (use -noTetCount if it's 
// not there).  The tet entries are assumed to be zero-based, unless you 
// specify -oneBasedIndexing.  And the SCIRun output file is written in 
// ASCII, unless you specify -binOutput.

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/NoData.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Init/init.h>
#include <StandAlone/convert/FileUtils.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool ptsCountHeader;
int baseIndex;
bool elementsCountHeader;
bool binOutput;
bool debugOn;

void setDefaults() {
  ptsCountHeader=true;
  baseIndex=0;
  elementsCountHeader=true;
  binOutput=false;
  debugOn=false;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      ptsCountHeader=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-noElementsCount")) {
      elementsCountHeader=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-oneBasedIndexing")) {
      baseIndex=1;
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
  cerr << "\n Usage: "<<progName<<" pts tets TetVolMesh [-noPtsCount] [-noElementsCount] [-oneBasedIndexing] [-binOutput] [-debug]\n\n";
  cerr << "\t This program will read in a .pts (specifying the x/y/z \n";
  cerr << "\t coords of each point, one per line, entries separated by \n";
  cerr << "\t white space, file can have an optional one line header \n";
  cerr << "\t specifying number of points... and if it doesn't, you have \n";
  cerr << "\t to use the -noPtsCount command-line argument) and a .tet \n";
  cerr << "\t file (specifying i/j/k/l indices for each tet, also one per \n";
  cerr << "\t line, again with an optional one line header (use \n";
  cerr << "\t -noTetCount if it's not there).  The tet entries are assumed \n";
  cerr << "\t to be zero-based, unless you specify -oneBasedIndexing.  And \n";
  cerr << "\t the SCIRun output file is written in ASCII, unless you \n";
  cerr << "\t specify -binOutput.\n\n";
}

int
main(int argc, char **argv) {

  if (argc < 4 || argc > 9) {
    printUsageInfo(argv[0]);
    return 2;
  }

  SCIRunInit();
  setDefaults();
  typedef TetVolMesh<TetLinearLgn<Point> > TVMesh;
  TVMesh *tvm = new TVMesh();

  char *ptsName = argv[1];
  char *tetsName = argv[2];
  char *fieldName = argv[3];
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
    tvm->add_point(Point(x,y,z));
    if (debugOn) 
      cerr << "Added point #"<<i<<": ("<<x<<", "<<y<<", "<<z<<")\n";
  }
  cerr << "done adding points.\n";

  int ntets;
  if (!elementsCountHeader) ntets = getNumNonEmptyLines(tetsName);
  ifstream tetsstream(tetsName);
  if (tetsstream.fail()) {
    cerr << "Error -- Could not open file " << tetsName << "\n";
    return 2;
  }
  if (elementsCountHeader) tetsstream >> ntets;
  cerr << "number of tets = "<< ntets <<"\n";
  for (i=0; i<ntets; i++) {
    int n1, n2, n3, n4;
    tetsstream >> n1 >> n2 >> n3 >> n4;
    n1-=baseIndex; 
    n2-=baseIndex; 
    n3-=baseIndex; 
    n4-=baseIndex; 
    if (n1<0 || n1>=npts) { 
      cerr << "Error -- n1 ("<<i<<") out of bounds: "<<n1<<"\n"; 
      return 2; 
    }
    if (n2<0 || n2>=npts) { 
      cerr << "Error -- n2 ("<<i<<") out of bounds: "<<n2<<"\n"; 
      return 2; 
    }
    if (n3<0 || n3>=npts) { 
      cerr << "Error -- n3 ("<<i<<") out of bounds: "<<n3<<"\n"; 
      return 2; 
    }
    if (n4<0 || n4>=npts) { 
      cerr << "Error -- n4 ("<<i<<") out of bounds: "<<n4<<"\n"; 
      return 2; 
    }
    tvm->add_tet(n1, n2, n3, n4);
    if (debugOn) 
      cerr << "Added tet #"<<i<<": ["<<n1<<" "<<n2<<" "<<n3<<" "<<n4<<"]\n";
  }
  cerr << "done adding elements.\n";

  typedef NoDataBasis<double>                DatBasis;
  typedef GenericField<TVMesh, DatBasis, vector<double> > TVField; 

  TVField *tv = scinew TVField(tvm);
  FieldHandle tvH(tv);
  
  if (binOutput) {
    BinaryPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, tvH);
  } else {
    TextPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, tvH);
  }
  return 0;  
}    
