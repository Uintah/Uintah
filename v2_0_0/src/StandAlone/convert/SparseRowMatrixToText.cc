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
 *  SparseRowMatrixToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun SparseRowMatrix, and will save
// it out to a text version: a .txt file.
// The .txt file will contain one data value per line, consisting of
// the row index, the column index, and the data value (white-space
// separated); the file will also have a one line header, specifying
// the number of rows and number of columns in the matrix, unless the
// user specifies the -noHeader command-line argument.
// The user can specify the -oneBasedIndexing flag to support fortran 
// and matlab type matrices; the default is zero-based indexing.

#include <Core/Datatypes/SparseRowMatrix.h>
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
int baseIndex;

void setDefaults() {
  header=true;
  baseIndex=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-oneBasedIndexing")) {
      baseIndex=1;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" SparseRowMatrix textfile [-noHeader]\n\n";
  cerr << "\t This program will read in a SCIRun SparseRowMatrix, and will \n";
  cerr << "\t save it out to a text version: a .txt file.  The .txt file \n";
  cerr << "\t will contain one data value per line, consisting of the row \n";
  cerr << "\t index, the column index, and the data value (white-space \n";
  cerr << "\t separated); the file will also have a one line header, \n";
  cerr << "\t specifying the number of rows and number of columns in the \n";
  cerr << "\t matrix, unless the user specifies the -noHeader command-line \n";
  cerr << "\t argument. \n";
  cerr << "\t The user can specify the -oneBasedIndexing flag to support \n";
  cerr << "\t fortran and matlab type matrices; the default is zero-based \n";
  cerr << "\t indexing.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 5) {
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
  SparseRowMatrix *srm = dynamic_cast<SparseRowMatrix *>(handle.get_rep());
  if (!srm) {
    cerr << "Error -- input field wasn't a SparseRowMatrix\n";
    exit(0);
  }

  FILE *fTxt = fopen(textfileName, "wt");
  if (!fTxt) {
    cerr << "Error -- couldn't open output file "<<textfileName<<"\n";
    exit(0);
  }
  int nr=srm->nrows();
  int nc=srm->ncols();
  double *a = srm->get_val();
  int *rows = srm->get_row();
  int *columns = srm->get_col();
  int nnz = srm->get_nnz();
  cerr << "Number of rows = "<<nr<<"  number of columns = "<<nc<<"  nnz = "<<nnz<<"\n";
  if (header) fprintf(fTxt, "%d %d %d\n", nr, nc, nnz);

  int idx=0;
  for (int r=0; r<nr; r++) {
    while(idx<rows[r+1]) {
      fprintf(fTxt, "%d %d %lf\n", r+baseIndex, columns[idx]+baseIndex, a[idx]);
      idx++;
    }
  }
  fclose(fTxt);
  return 0;  
}
