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
 *  RawToDenseMatrix.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int binOutput;
int debugOn;
int groundZero;

void setDefaults() {
  binOutput=0;
  debugOn=0;
  groundZero=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg], "-binOutput")) {
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

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 6) {
    cerr << "Usage: "<<argv[0]<<" data DenseMatrix [-binOutput] [-debug] [-groundZero]\n";
    return 0;
  }
  setDefaults();

  char *dataName = argv[1];
  char *DenseMatrixName = argv[2];
  if (!parseArgs(argc, argv)) return 0;

  ifstream datastream(dataName);
  int nr, nc;
  datastream >> nr >> nc;
  DenseMatrix *dm = scinew DenseMatrix(nr, nc);
  cerr << "Matrix size: "<<nr<<" x "<<nc<<"\n";
  int r,c;
  double x;
  for (r=0; r<nr; r++) {
    for (c=0; c<nc; c++) {
      datastream >> x;
      if (groundZero && r!=0) x-= (*dm)[0][c];
      (*dm)[r][c] = x;
    }
  }
  if (groundZero) for (c=0; c<nc; c++) (*dm)[0][c]=0;
  cerr << "Done reading data.\n";

  MatrixHandle dmH(dm);

  if (binOutput) {
    BinaryPiostream out_stream(DenseMatrixName, Piostream::Write);
    Pio(out_stream, dmH);
  } else {
    TextPiostream out_stream(DenseMatrixName, Piostream::Write);
    Pio(out_stream, dmH);
  }
  return 0;  
}    
