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
 *  TriSurfFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun TriSurfField, and will save
// out the TriSurfMesh into two files: a .pts file and a .fac file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying number of points, unless the
// user specifies the -noPtsCount command-line argument.
// The .tri file will specify the i/j/k indices for each triangle, 
// also one per line, again with a one line header (unless a 
// -noElementsCount flag is used).  The tri entries will be zero-based, 
// unless the user specifies -oneBasedIndexing.

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
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
  cerr << "\n Usage: "<<progName<<" TriSurfField pts tris [-noPtsCount] [-noElementsCount] [-oneBasedIndexing]\n\n";
  cerr << "\t This program will read in a SCIRun TriSurfField, and will \n";
  cerr << "\t save out the TriSurfMesh into two files: a .pts file and a \n";
  cerr << "\t .fac file.  The .pts file will specify the x/y/z coordinates \n";
  cerr << "\t of each point, one per line, entries separated by white \n";
  cerr << "\t space; the file will also have a one line header, specifying \n";
  cerr << "\t number of points, unless the user specifies the -noPtsCount \n";
  cerr << "\t command-line argument.  The .tri file will specify the i/j/k \n";
  cerr << "\t indices for each triangle, also one per line, again with a \n";
  cerr << "\t one line header (unless a -noElementsCount flag is used).  The \n";
  cerr << "\t tri entries will be zero-based, unless the user specifies \n";
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
  char *trisName = argv[3];
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
  if (handle->get_type_description(1)->get_name().find("TriSurfField") !=
      string::npos)
  {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="
	 << handle->get_type_description(1)->get_name() << std::endl;
    return 2;
  }

  typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;

  MeshHandle mH = handle->mesh();
  TSMesh *tsm = dynamic_cast<TSMesh *>(mH.get_rep());
  TSMesh::Node::iterator niter; 
  TSMesh::Node::iterator niter_end; 
  TSMesh::Node::size_type nsize; 
  tsm->begin(niter);
  tsm->end(niter_end);
  tsm->size(nsize);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    return 2;
  }
  if (ptsCountHeader) fprintf(fPts, "%d\n", (unsigned)(nsize));
  cerr << "Number of points = "<< nsize <<"\n";
  while(niter != niter_end) {
    Point p;
    tsm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);

  TSMesh::Face::size_type fsize; 
  TSMesh::Face::iterator fiter; 
  TSMesh::Face::iterator fiter_end; 
  TSMesh::Node::array_type fac_nodes(3);
  tsm->size(fsize);
  tsm->begin(fiter);
  tsm->end(fiter_end);
  FILE *fTris = fopen(trisName, "wt");
  if (!fTris) {
    cerr << "Error opening output file "<<trisName<<"\n";
    return 2;
  }
  if (elementsCountHeader) fprintf(fTris, "%d\n", (unsigned)(fsize));
  cerr << "Number of tris = "<< fsize <<"\n";
  while(fiter != fiter_end) {
    tsm->get_nodes(fac_nodes, *fiter);
    fprintf(fTris, "%d %d %d\n", 
	    (int)fac_nodes[0]+baseIndex,
	    (int)fac_nodes[1]+baseIndex,
	    (int)fac_nodes[2]+baseIndex);
    ++fiter;
  }
  fclose(fTris);

  return 0;  
}    
