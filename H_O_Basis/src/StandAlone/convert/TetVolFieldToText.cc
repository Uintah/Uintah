/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
// -noElementsCount flag is used).  The tet entries will be zero-based, 
// unless the user specifies -oneBasedIndexing.

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
#include <Core/Init/init.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool ptsCountHeader;
int baseIndex;
bool elementsCountHeader;

void setDefaults() {
  ptsCountHeader=true;
  baseIndex=0;
  elementsCountHeader=true;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      ptsCountHeader=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-noElementsCount")) {
      elementsCountHeader=false;
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
  cerr << "\n Usage: "<<progName<<" TetVolField pts tets [-noPtsCount] [-noElementsCount] [-oneBasedIndexing]\n\n";
  cerr << "\t This program will read in a SCIRun TetVolField, and will \n";
  cerr << "\t save out the TetVolMesh into two files: a .pts file and a \n";
  cerr << "\t .tet file.  The .pts file will specify the x/y/z coordinates \n";
  cerr << "\t of each point, one per line, entries separated by white \n";
  cerr << "\t space; the file will also have a one line header, specifying \n";
  cerr << "\t number of points, unless the user specifies the -noPtsCount \n";
  cerr << "\t command-line argument.  The .tet file will specify the \n";
  cerr << "\t i/j/k/l indices for each tet, also one per line, again with \n";
  cerr << "\t a one line header (unless a -noElementsCount flag is used).  The \n";
  cerr << "\t tet entries will be zero-based, unless the user specifies \n";
  cerr << "\t -oneBasedIndexing.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 4 || argc > 7) {
    printUsageInfo(argv[0]);
    return 2;
  }
  SCIRunInit();
  setDefaults();

  char *fieldName = argv[1];
  char *ptsName = argv[2];
  char *tetsName = argv[3];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 2;
  }

  FieldHandle handle;
  Piostream* stream=auto_istream(fieldName);
  if (!stream) {
    cerr << "Couldn't open file "<<fieldName<<".  Exiting...\n";
    return 2;
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<fieldName<<".  Exiting...\n";
    return 2;
  }
  if (handle->get_type_description(1)->get_name().find("TetVolField") !=
      string::npos)
  {
    cerr << "Error -- input field wasn't a TetVolField (type_name="
	 << handle->get_type_description(1)->get_name() << std::endl;
    return 2;
  }
  typedef TetVolMesh<TetLinearLgn<Point> > TVMesh;

  MeshHandle mH = handle->mesh();
  TVMesh *tvm = dynamic_cast<TVMesh *>(mH.get_rep());
  TVMesh::Node::iterator niter; 
  TVMesh::Node::iterator niter_end; 
  TVMesh::Node::size_type nsize; 
  tvm->begin(niter);
  tvm->end(niter_end);
  tvm->size(nsize);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    return 2;
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

  TVMesh::Cell::size_type csize; 
  TVMesh::Cell::iterator citer; 
  TVMesh::Cell::iterator citer_end; 
  TVMesh::Node::array_type cell_nodes;
  tvm->size(csize);
  tvm->begin(citer);
  tvm->end(citer_end);
  FILE *fTets = fopen(tetsName, "wt");
  if (!fTets) {
    cerr << "Error opening output file "<<tetsName<<"\n";
    return 2;
  }
  size=(unsigned)(csize);
  if (elementsCountHeader) fprintf(fTets, "%d\n", size);
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
