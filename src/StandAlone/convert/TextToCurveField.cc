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
 *  TextToCurveField.cc
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
// .edge file (specifying i/j indices for each edge, also one per 
// line, again with an optional one line header (use -noElementsCount if it's 
// not there).  The edge entries are assumed to be zero-based, unless you 
// specify -oneBasedIndexing.  And the SCIRun output file is written in 
// ASCII, unless you specify -binOutput.

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/CurveMesh.h>
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
  cerr << "\n Usage: "<<progName<<" pts edges CurveMesh [-noPtsCount] [-noElementsCount] [-oneBasedIndexing] [-binOutput] [-debug]\n\n";
  cerr << "\t This program will read in a .pts (specifying the x/y/z \n";
  cerr << "\t coords of each point, one per line, entries separated by \n";
  cerr << "\t white space, file can have an optional one line header \n";
  cerr << "\t specifying number of points... and if it doesn't, you have \n";
  cerr << "\t to use the -noPtsCount command-line argument) and a .edge \n";
  cerr << "\t file (specifying i/j indices for each edge, also one per \n";
  cerr << "\t line, again with an optional one line header (use \n";
  cerr << "\t -noElementsCount if it's not there).  The edge entries are \n";
  cerr << "\t assumed to be zero-based, unless you specify \n";
  cerr << "\t -oneBasedIndexing.  And the SCIRun output file is written in \n";
  cerr << "\t ASCII, unless you specify -binOutput.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 4 || argc > 9) {
    printUsageInfo(argv[0]);
    return 2;
  }
  SCIRunInit();
  setDefaults();
  typedef CurveMesh<CrvLinearLgn<Point> > CMesh;

  CMesh *cm = new CMesh();
  char *ptsName = argv[1];
  char *edgesName = argv[2];
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
    cm->add_point(Point(x,y,z));
    if (debugOn) 
      cerr << "Added point #"<<i<<": ("<<x<<", "<<y<<", "<<z<<")\n";
  }
  cerr << "done adding points.\n";

  int nedges;
  if (!elementsCountHeader) nedges = getNumNonEmptyLines(edgesName);
  ifstream edgesstream(edgesName);
  if (edgesstream.fail()) {
    cerr << "Error -- Could not open file " << edgesName << "\n";
    return 2;
  }
  if (elementsCountHeader) edgesstream >> nedges;
  cerr << "number of edges = "<< nedges <<"\n";
  for (i=0; i<nedges; i++) {
    int n1, n2;
    edgesstream >> n1 >> n2;
    n1-=baseIndex; 
    n2-=baseIndex; 
    if (n1<0 || n1>=npts) { 
      cerr << "Error -- n1 ("<<i<<") out of bounds: "<<n1<<"\n"; 
      return 2; 
    }
    if (n2<0 || n2>=npts) { 
      cerr << "Error -- n2 ("<<i<<") out of bounds: "<<n2<<"\n"; 
      return 2; 
    }
    cm->add_edge(n1, n2);
    if (debugOn) 
      cerr << "Added edge #"<<i<<": ["<<n1<<" "<<n2<<"]\n";
  }
  cerr << "done adding edges.\n";

  typedef NoDataBasis<double>                DatBasis;
  typedef GenericField<CMesh, DatBasis, vector<double> > CField;
     
  CField *cf = scinew CField(cm);

  FieldHandle cfH(cf);
  
  if (binOutput) {
    BinaryPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, cfH);
  } else {
    TextPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, cfH);
  }
  return 0;  
}    
