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
 *  TextToDenseMatrix.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a .txt file specifying a matrix.  The .txt
// file will have white-space separated matrix entries and a one line 
// header indicating the number of rows and number of columns of the 
// matrix (unless the user has specified the -noHeader flag, in which 
// case the next two command line arguments must be the number of rows 
// and the number of columns in the matrix).
// The SCIRun .mat file will be saved as ASCII by default, unless the
// user specified the -binOutput flag.

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
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
int nr;
int nc;

void setDefaults() {
  header=true;
  binOutput=false;
  debugOn=false;
  nr=0;
  nc=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
      currArg++;
      nr=atoi(argv[currArg]);
      currArg++;
      nc=atoi(argv[currArg]);
      currArg++;
    } else if (!strcmp(argv[currArg],"-binOutput")) {
      binOutput=true;
      currArg++;
    } else if (!strcmp(argv[currArg],"-debugOn")) {
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
  cerr << "\n Usage: "<<progName<<" textfile DenseMatix [-noHeader nrows ncols] [-binOutput] [-debugOn]\n\n";
  cerr << "\t This program will read in a .txt file specifying a matrix. \n";
  cerr << "\t The .txt file will have white-space separated matrix entries \n";
  cerr << "\t and  a one line header indicating the number of rows and \n";
  cerr << "\t number of columns of the matrix (unless the user has \n";
  cerr << "\t specified the -noHeader flag, in which case the next two \n";
  cerr << "\t command line arguments must be the number of rows and the \n";
  cerr << "\t number of columns in the matrix).  The SCIRun .mat file will \n";
  cerr << "\t be saved as ASCII by default, unless the user specified the \n";
  cerr << "\t -binOutput flag.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 8) {
    printUsageInfo(argv[0]);
    return 0;
  }
  setDefaults();

  char *textfileName = argv[1];
  char *matrixName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }

  ifstream matstream(textfileName);
  if (header) matstream >> nr >> nc;

  cerr << "nrows="<<nr<<" ncols="<<nc<<"\n";
  DenseMatrix *dm = scinew DenseMatrix(nr,nc);

  int r, c;
  for (r=0; r<nr; r++)
    for (c=0; c<nc; c++) {
      double d;
      matstream >> d;
      (*dm)[r][c]=d;
      if (debugOn) 
	cerr << "matrix["<<r<<"]["<<c<<"]="<<d<<"\n";
    }
  cerr << "done building matrix.\n";

  MatrixHandle mH(dm);

  if (binOutput) {
    BinaryPiostream out_stream(matrixName, Piostream::Write);
    Pio(out_stream, mH);
  } else {
    TextPiostream out_stream(matrixName, Piostream::Write);
    Pio(out_stream, mH);
  }
  return 0;  
}    
