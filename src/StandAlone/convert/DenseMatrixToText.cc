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
 *  DenseMatrixToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun DenseMatrix, and will save
// it out to a text version: a .txt file.
// The .txt file will contain one data value per line; the file will
// also have a one line header, specifying the number of rows and number
// of columns in the matrix, unless the user specifies the -noHeader
// command-line argument.

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <StandAlone/convert/FileUtils.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool header;

void setDefaults() {
  header=true;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" DenseMatrix textfile [-noHeader]\n\n";
  cerr << "\t This program will read in a SCIRun DenseMatrix, and will \n";
  cerr << "\t save it out to a text version: a .txt file. \n";
  cerr << "\t The .txt file will contain one data value per line; the \n";
  cerr << "\t file will also have a one line header, specifying the \n";
  cerr << "\t number of rows and number of columns in the matrix, unless \n";
  cerr << "\t the user specifies the -noHeader command-line argument.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    printUsageInfo(argv[0]);
    return 0;
  }
  setDefaults();

  char *matrixName = argv[1];
  char *textfileName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }

  MatrixHandle handle;
  Piostream* stream=auto_istream(matrixName);
  if (!stream) {
    cerr << "Couldn't open file "<<matrixName<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading matrix from file "<<matrixName<<".  Exiting...\n";
    exit(0);
  }
  DenseMatrix *dm = dynamic_cast<DenseMatrix *>(handle.get_rep());
  if (!dm) {
    cerr << "Error -- input field wasn't a DenseMatrix\n";
    exit(0);
  }

  int nr=dm->nrows();
  int nc=dm->ncols();
  cerr << "Number of rows = "<<nr<<"  number of columns = "<<nc<<"\n";
  FILE *fTxt = fopen(textfileName, "wt");
  if (!fTxt) {
    cerr << "Error -- couldn't open output file "<<textfileName<<"\n";
    exit(0);
  }
  if (header) fprintf(fTxt, "%d %d\n", nr, nc);
  for (int r=0; r<nr; r++) {
    for (int c=0; c<nc; c++)
      fprintf(fTxt, "%lf ", (*dm)[r][c]);
    fprintf(fTxt, "\n");
  }
  fclose(fTxt);
  return 0;  
}    
