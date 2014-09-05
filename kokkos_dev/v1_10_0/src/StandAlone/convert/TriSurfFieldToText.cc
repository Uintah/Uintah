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

#include <Core/Datatypes/TriSurfField.h>
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
    return 0;
  }
  setDefaults();

  char *fieldName = argv[1];
  char *ptsName = argv[2];
  char *trisName = argv[3];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }

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
  if (handle->get_type_description(0)->get_name() != "TriSurfField") {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<handle->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mH = handle->mesh();
  TriSurfMesh *tsm = dynamic_cast<TriSurfMesh *>(mH.get_rep());
  TriSurfMesh::Node::iterator niter; 
  TriSurfMesh::Node::iterator niter_end; 
  TriSurfMesh::Node::size_type nsize; 
  tsm->begin(niter);
  tsm->end(niter_end);
  tsm->size(nsize);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    exit(0);
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

  TriSurfMesh::Face::size_type fsize; 
  TriSurfMesh::Face::iterator fiter; 
  TriSurfMesh::Face::iterator fiter_end; 
  TriSurfMesh::Node::array_type fac_nodes(3);
  tsm->size(fsize);
  tsm->begin(fiter);
  tsm->end(fiter_end);
  FILE *fTris = fopen(trisName, "wt");
  if (!fTris) {
    cerr << "Error opening output file "<<trisName<<"\n";
    exit(0);
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
