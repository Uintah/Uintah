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
 *  TextToColumnMatrix.cc
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
// file will have whitespace-separated matrix entries and a one line
// header indicating the number of rows and number of columns of the
// matrix (unless the user has specified the -noHeader flag.
// The SCIRun .mat file will be saved as ASCII by default, unless the
// user specified the -binOutput flag.

#include <Core/Datatypes/ColumnMatrix.h>
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

void setDefaults() {
  header=true;
  binOutput=false;
  debugOn=false;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
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
  cerr << "\n Usage: "<<progName<<" textfile ColumnMatix [-noHeader] [-binOutput] [-debugOn]\n\n";
  cerr << "\t This program will read in a .txt file specifying a matrix. \n";
  cerr << "\t The .txt file will have whitespace-separated matrix entries \n";
  cerr << "\t and a one line header indicating the number of rows and \n";
  cerr << "\t number of columns of the matrix (unless the user has \n";
  cerr << "\t specified the -noHeader flag.  The SCIRun .mat file will be \n";
  cerr << "\t saved as ASCII by default, unless the user specified the \n";
  cerr << "\t -binOutput flag.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 6) {
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

  int nr;
  if (!header) nr = getNumNonEmptyLines(textfileName);
  ifstream matstream(textfileName);
  if (header) matstream >> nr;
  cerr << "nrows="<<nr<<"\n";
  ColumnMatrix *cm = scinew ColumnMatrix(nr);
  int r;
  for (r=0; r<nr; r++) {
    double d;
    matstream >> d;
    (*cm)[r]=d;
    if (debugOn) 
      cerr << "matrix["<<r<<"]="<<d<<"\n";
    }
  cerr << "done building matrix.\n";

  MatrixHandle mH(cm);

  if (binOutput) {
    BinaryPiostream out_stream(matrixName, Piostream::Write);
    Pio(out_stream, mH);
  } else {
    TextPiostream out_stream(matrixName, Piostream::Write);
    Pio(out_stream, mH);
  }
  return 0;  
}    
