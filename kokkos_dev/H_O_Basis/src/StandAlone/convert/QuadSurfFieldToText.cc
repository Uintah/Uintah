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
 *  QuadSurfFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun QuadSurfField, and will save
// out the QuadSurfMesh into two files: a .pts file and a .quad file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying number of points, unless the
// user specifies the -noPtsCount command-line argument.
// The .quad file will specify the i/j/k/l indices for each quad, 
// also one per line, again with a one line header (unless a 
// -noElementsCount flag is used).  The quad entries will be zero-based, 
// unless the user specifies -oneBasedIndexing.

#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
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
  cerr << "\n Usage: "<<progName<<" QuadSurfField pts quads [-noPtsCount] [-noElementsCount] [-oneBasedIndexing]\n\n";
  cerr << "\t This program will read in a SCIRun QuadSurfField, and will \n";
  cerr << "\t save out the QuadSurfMesh into two files: a .pts file and a \n";
  cerr << "\t .quad file. \n";
  cerr << "\t The .pts file will specify the x/y/z coordinates of each \n";
  cerr << "\t point, one per line, entries separated by white space; the \n";
  cerr << "\t file will also have a one line header, specifying number of \n";
  cerr << "\t points, unless the user specifies the -noPtsCount \n";
  cerr << "\t command-line argument.  The .quad file will specify the \n";
  cerr << "\t i/j/k/l indices for each quad, also one per line, again with \n";
  cerr << "\t a one line header (unless a -noElementsCount flag is used). \n";
  cerr << "\t The quad entries will be zero-based, unless the user \n";
  cerr << "\t specifies -oneBasedIndexing.\n\n";
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
  char *quadsName = argv[3];
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
  if (handle->get_type_description(1)->get_name().find("QuadSurfField") != 
      string::npos) 
  {
    cerr << "Error -- input field wasn't a QuadSurfField (type_name="
	 << handle->get_type_description(1)->get_name() << std::endl;
    return 2;
  }
  typedef QuadSurfMesh<QuadBilinearLgn<Point> > QSMesh;
  MeshHandle mH = handle->mesh();
  QSMesh *qsm = dynamic_cast<QSMesh *>(mH.get_rep());
  QSMesh::Node::iterator niter; 
  QSMesh::Node::iterator niter_end; 
  QSMesh::Node::size_type nsize; 
  qsm->begin(niter);
  qsm->end(niter_end);
  qsm->size(nsize);
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
    qsm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);

  QSMesh::Face::size_type fsize; 
  QSMesh::Face::iterator fiter; 
  QSMesh::Face::iterator fiter_end; 
  QSMesh::Node::array_type face_nodes(4);
  qsm->size(fsize);
  qsm->begin(fiter);
  qsm->end(fiter_end);
  FILE *fQuads = fopen(quadsName, "wt");
  if (!fQuads) {
    cerr << "Error opening output file "<<quadsName<<"\n";
    return 2;
  }
  size=(unsigned)(fsize);
  if (elementsCountHeader) fprintf(fQuads, "%d\n", size);
  cerr << "Number of quads = "<< fsize <<"\n";
  while(fiter != fiter_end) {
    qsm->get_nodes(face_nodes, *fiter);
    fprintf(fQuads, "%d %d %d %d\n", 
	    (int)face_nodes[0]+baseIndex,
	    (int)face_nodes[1]+baseIndex,
	    (int)face_nodes[2]+baseIndex,
	    (int)face_nodes[3]+baseIndex);
    ++fiter;
  }
  fclose(fQuads);

  return 0;  
}    
