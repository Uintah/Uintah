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
 *  TetVolFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun TetVolField, and will save
// out the TetVolMesh into two files: a .pts file and a .tet file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying number of points, unless the
// user specifies the -noPtsCount command-line argument.
// The .tet file will specify the i/j/k/l indices for each tet, 
// also one per line, again with a one line header (unless a 
// -noTetsCount flag is used).  The tet entries will be zero-based, 
// unless the user specifies -oneBasedIndexing.

#include <Core/Datatypes/TetVolField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool ptsCountHeader;
int baseIndex;
bool tetsCountHeader;

void setDefaults() {
  ptsCountHeader=true;
  baseIndex=0;
  tetsCountHeader=true;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      ptsCountHeader=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-noTetsCount")) {
      tetsCountHeader=false;
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
  if (argc < 4 || argc > 7) {
    cerr << "Usage: "<<argv[0]<<" TetVolField pts tets [-noPtsCount] [-noTetsCount] [-oneBasedIndexing]\n";
    return 0;
  }
  setDefaults();

  char *fieldName = argv[1];
  char *ptsName = argv[2];
  char *tetsName = argv[3];
  if (!parseArgs(argc, argv)) return 0;

  FieldHandle handle;
  Piostream* stream=auto_istream(fieldName);
  if (!stream) {
    cerr << "Couldn't open file "<<fieldName<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<fieldName<<".  Exiting...\n";
    exit(0);
  }
  if (handle->get_type_description(0)->get_name() != "TetVolField") {
    cerr << "Error -- input field wasn't a TetVolField (type_name="<<handle->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mH = handle->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh *>(mH.get_rep());
  TetVolMesh::Node::iterator niter; 
  TetVolMesh::Node::iterator niter_end; 
  TetVolMesh::Node::size_type nsize; 
  tvm->begin(niter);
  tvm->end(niter_end);
  tvm->size(nsize);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    exit(0);
  }
  int size=(unsigned)(nsize);
  if (ptsCountHeader) fprintf(fPts, "%d\n", size);
  cerr << "Number of points = "<< nsize <<"\n";
  while(niter != niter_end) {
    Point p;
    tvm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);

  TetVolMesh::Cell::size_type csize; 
  TetVolMesh::Cell::iterator citer; 
  TetVolMesh::Cell::iterator citer_end; 
  TetVolMesh::Node::array_type cell_nodes(4);
  tvm->size(csize);
  tvm->begin(citer);
  tvm->end(citer_end);
  FILE *fTets = fopen(tetsName, "wt");
  if (!fTets) {
    cerr << "Error opening output file "<<tetsName<<"\n";
    exit(0);
  }
  size=(unsigned)(csize);
  if (tetsCountHeader) fprintf(fTets, "%d\n", size);
  cerr << "Number of tets = "<< csize <<"\n";
  while(citer != citer_end) {
    tvm->get_nodes(cell_nodes, *citer);
    fprintf(fTets, "%d %d %d %d\n", 
	    (int)cell_nodes[0]+baseIndex,
	    (int)cell_nodes[1]+baseIndex,
	    (int)cell_nodes[2]+baseIndex,
	    (int)cell_nodes[3]+baseIndex);
    ++citer;
  }
  fclose(fTets);

  return 0;  
}    
