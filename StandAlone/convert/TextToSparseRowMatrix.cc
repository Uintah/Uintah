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
 *  TextToSparseRowMatrix.cc
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
// file will have one matrix entry per line (each line will consist of
// a row index, a column index, and a data value, all white-space 
// separated), and will have a one line header indicating the number of
// rows, number of columns, and number of entries in the matrix (unless
// the user has specified the -noHeader flag, in which case the next three 
// command line arguments must be the number of rows, number of columns,
// and number of entries in the matrix).
// The SCIRun .mat file will be saved as ASCII by default, unless the
// user specified the -binOutput flag.

#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

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
int nnz;

void setDefaults() {
  header=true;
  binOutput=false;
  debugOn=false;
  nr=0;
  nc=0;
  nnz=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
      currArg++;
      nr=atoi(argv[currArg]);
      currArg++;
      nc=atoi(argv[currArg]);
      currArg++;
      nnz=atoi(argv[currArg]);
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
  if (argc < 3 || argc > 8) {
    cerr << "Usage: "<<argv[0]<<" textfile SparseRowMatix [-noHeader nrows ncols] [-binOutput] [-debugOn]\n";
    return 0;
  }
  setDefaults();

  char *textfileName = argv[1];
  char *matrixName = argv[2];
  if (!parseArgs(argc, argv)) return 0;
  ifstream matstream(textfileName);
  if (!header) matstream >> nr >> nc >> nnz;

  cerr << "nrows="<<nr<<" ncols="<<nc<<" # of non-zeros="<<nnz<<"\n";

  int *columns = new int[nnz];
  int *rows = new int[nr+1];
  double *a = new double[nnz];

  int r, c;
  double d;
  int e;
  int last_r=-1;
  for (e=0; e<nnz; e++) {
    matstream >> r >> c >> d;
    if (debugOn)
      cerr << "matrix["<<r<<"]["<<c<<"]="<<d<<"\n";
    columns[e]=c;
    a[e]=d;
    while(r<last_r) {
      last_r++;
      rows[r]=e;
    }
  }
  SparseRowMatrix *srm = scinew SparseRowMatrix(nr,nc, rows, columns, nnz, a);

  cerr << "Done building matrix.\n";

  MatrixHandle mH(srm);

  if (binOutput) {
    BinaryPiostream out_stream(matrixName, Piostream::Write);
    Pio(out_stream, mH);
  } else {
    TextPiostream out_stream(matrixName, Piostream::Write);
    Pio(out_stream, mH);
  }
  return 0;  
}    
