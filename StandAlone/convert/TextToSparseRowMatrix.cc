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
// The user can specify the -oneBasedIndexing flag to support fortran
// and matlab type matrices; the default is zero-based indexing.
// Matrix entries must have ascending row indices, and for rows with
// multiple entries, their column indices must be in ascending order.
// The SCIRun .mat file will be saved as ASCII by default, unless the
// user specified the -binOutput flag.

#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
#if defined(__APPLE__)
#  include <Core/Datatypes/MacForceLoad.h>
#endif
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
int baseIndex;

void setDefaults() {
  header=true;
  binOutput=false;
  debugOn=false;
  nr=0;
  nc=0;
  nnz=0;
  baseIndex=0;
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
      nnz=atoi(argv[currArg]);
      currArg++;
    } else if (!strcmp(argv[currArg],"-binOutput")) {
      binOutput=true;
      currArg++;
    } else if (!strcmp(argv[currArg],"-oneBasedIndexing")) {
      baseIndex=1;
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
  cerr << "\n Usage: "<<progName<<" textfile SparseRowMatix [-noHeader nrows ncols nnz] [-binOutput] [-debugOn]\n\n";
  cerr << "\t This program will read in a .txt file specifying a matrix. \n";
  cerr << "\t The .txt file will have one matrix entry per line (each \n";
  cerr << "\t line will consist of a row index, a column index, and a \n";
  cerr << "\t data value, all white-space separated), and will have a one \n";
  cerr << "\t line header indicating the number of rows, number of \n";
  cerr << "\t columns, and number of entries in the matrix (unless the \n";
  cerr << "\t user has specified the -noHeader flag, in which case the \n";
  cerr << "\t next three command line arguments must be the number of \n";
  cerr << "\t rows, number of columns, and number of entries in the \n";
  cerr << "\t matrix). \n";
  cerr << "\t The user can specify the -oneBasedIndexing flag to support \n";
  cerr << "\t fortran and matlab type matrices; the default is zero-based \n";
  cerr << "\t indexing. \n";
  cerr << "\t Matrix entries must have ascending row indices, and for rows \n";
  cerr << "\t with multiple entries, their column indices must be in \n";
  cerr << "\t ascending order. \n";
  cerr << "\t The SCIRun .mat file will be saved as ASCII by default, \n";
  cerr << "\t unless the user specified the -binOutput flag.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 9) {
    printUsageInfo(argv[0]);
    return 0;
  }
#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  setDefaults();

  char *textfileName = argv[1];
  char *matrixName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }

  ifstream matstream(textfileName);
  if (header) matstream >> nr >> nc >> nnz;

  cerr << "nrows="<<nr<<" ncols="<<nc<<" # of non-zeros="<<nnz<<"\n";

  int *columns = new int[nnz];
  int *rows = new int[nr+1];
  double *a = new double[nnz];

  int r, c;
  double d;
  int e;
  int last_r=-1;
  int lineno = 1;
  for (e=0; e<nnz; e++) {
    matstream >> r >> c >> d;
    if (debugOn)
      cerr << "matrix["<<r<<"]["<<c<<"]="<<d<<"\n";
    r-=baseIndex;
    c-=baseIndex;
    if (r < 0 || r >= nr)
    {
      cerr << "Error: Row " << r << " at entry " << lineno << 
	" is out of range.\n";
      return 0;
    }
    if (c < 0 || c >= nc)
    {
      cerr << "Error: Column " << c << " at entry " << lineno << 
	" is out of range.\n";
      return 0;
    }
    columns[e]=c;
    a[e]=d;
    while(last_r<r) {
      last_r++;
      rows[last_r]=e;
    }
    ++lineno;
  }
  while(last_r<nr) {
    last_r++;
    rows[last_r]=nnz;
  }

  SparseRowMatrix *srm = scinew SparseRowMatrix(nr, nc, rows, columns, nnz, a);

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
