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

#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Persistent/Pstreams.h>
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
  int currArg = 4;
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
  if (argc < 3 || argc > 4) {
    cerr << "Usage: "<<argv[0]<<" SparseRowMatrix textfile [-noHeader]\n";
    return 0;
  }

  char *matrixName = argv[1];
  char *textfileName = argv[2];
  if (!parseArgs(argc, argv)) return 0;

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
      fprintf(fTxt, "%d %d %lf\n", r, columns[idx], a[idx]);
      idx++;
    }
  }
  fclose(fTxt);
  return 0;  
}
