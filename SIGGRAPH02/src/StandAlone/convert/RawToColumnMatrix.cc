/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distetbuted under the License is distetbuted on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  RawToColumnMatrix.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int numRowsHeader;
int binOutput;
int debugOn;
int groundZero;

void setDefaults() {
  numRowsHeader=1;
  binOutput=0;
  debugOn=0;
  groundZero=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg], "-noNumRows")) {
      numRowsHeader=0;
      currArg++;
    } else if (!strcmp(argv[currArg], "-binOutput")) {
      binOutput=1;
      currArg++;
    } else if (!strcmp(argv[currArg], "-debug")) {
      debugOn=1;
      currArg++;
    } else if (!strcmp(argv[currArg], "-groundZero")) {
      groundZero=1;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

int getNumNonEmptyLines(char *fname) {
  // read through the file -- when you see a non-white-space set a flag to one.
  // when you get to the end of the line (or EOF), see if the flag has
  // been set.  if it has, increment the count and reset the flag to zero.

  FILE *fin = fopen(fname, "rt");
  int count=0;
  int haveNonWhiteSpace=0;
  int c;
  while ((c=fgetc(fin)) != EOF) {
    if (!isspace(c)) haveNonWhiteSpace=1;
    else if (c=='\n' && haveNonWhiteSpace) {
      count++;
      haveNonWhiteSpace=0;
    }
  }
  if (haveNonWhiteSpace) count++;
  cerr << "number of nonempty lines was: "<<count<<"\n";
  return count;
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 7) {
    cerr << "Usage: "<<argv[0]<<" data ColumnMatrix [-noNumRows] [-binOutput] [-debug] [-groundZero]\n";
    return 0;
  }
  setDefaults();

  char *dataName = argv[1];
  char *columnMatrixName = argv[2];
  if (!parseArgs(argc, argv)) return 0;

  int ndata;
  if (!numRowsHeader) ndata = getNumNonEmptyLines(dataName);
  ifstream datastream(dataName);
  if (numRowsHeader) datastream >> ndata;
  cerr << "number of data values = "<< ndata <<"\n";
  int i;
  ColumnMatrix *cm = scinew ColumnMatrix(ndata);
  for (i=0; i<ndata; i++) {
    double x;
    datastream >> x;
    if (groundZero && i!=0) x-=(*cm)[0];
    (*cm)[i]=x;
  }
  if (groundZero) (*cm)[0]=0;

  cerr << "done adding data.\n";

  MatrixHandle cmh(cm);

  if (binOutput) {
    BinaryPiostream out_stream(columnMatrixName, Piostream::Write);
    Pio(out_stream, cmh);
  } else {
    TextPiostream out_stream(columnMatrixName, Piostream::Write);
    Pio(out_stream, cmh);
  }
  return 0;  
}
