/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software disedgebuted under the License is disedgebuted on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  CurveFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun CurveField, and will save
// out the CurveMesh into two files: a .pts file and a .edge file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, enedgees separated by white space; the file will
// also have a one line header, specifying number of points, unless the
// user specifies the -noPtsCount command-line argument.
// The .edge file will specify the i/j indices for each edge, 
// also one per line, again with a one line header (unless a 
// -noElementsCount flag is used).  The edge indicies will be zero-based, 
// unless the user specifies -oneBasedIndexing.

#include <Core/Datatypes/CurveField.h>
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
  cerr << "\n Usage: "<<progName<<" CurveField pts edges [-noPtsCount] [-noElementsCount] [-oneBasedIndexing]\n\n";
  cerr << "\t This program will read in a SCIRun CurveField, and will save \n";
  cerr << "\t out the CurveMesh into two files: a .pts file and a .edge \n";
  cerr << "\t file.  The .pts file will specify the x/y/z coordinates of \n";
  cerr << "\t each point, one per line, enedgees separated by white space; \n";
  cerr << "\t the file will also have a one line header, specifying number \n";
  cerr << "\t of points, unless the user specifies the -noPtsCount \n";
  cerr << "\t command-line argument.  The .edge file will specify the i/j \n";
  cerr << "\t indices for each edge, also one per line, again with a one \n";
  cerr << "\t line header (unless a -noElementsCount flag is used).  The edge \n";
  cerr << "\t indicies will be zero-based, unless the user specifies \n";
  cerr << "\t -oneBasedIndexing.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 4 || argc > 7) {
    printUsageInfo(argv[0]);
    return 0;
  }
#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  setDefaults();

  char *fieldName = argv[1];
  char *ptsName = argv[2];
  char *edgesName = argv[3];
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
  if (handle->get_type_description(0)->get_name() != "CurveField") {
    cerr << "Error -- input field wasn't a CurveField (type_name="<<handle->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mH = handle->mesh();
  CurveMesh *cm = dynamic_cast<CurveMesh *>(mH.get_rep());
  CurveMesh::Node::iterator niter; 
  CurveMesh::Node::iterator niter_end; 
  CurveMesh::Node::size_type nsize; 
  cm->begin(niter);
  cm->end(niter_end);
  cm->size(nsize);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    exit(0);
  }
  if (ptsCountHeader) fprintf(fPts, "%d\n", (unsigned)(nsize));
  cerr << "Number of points = "<< nsize <<"\n";
  while(niter != niter_end) {
    Point p;
    cm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);

  CurveMesh::Edge::size_type esize; 
  CurveMesh::Edge::iterator eiter; 
  CurveMesh::Edge::iterator eiter_end; 
  CurveMesh::Node::array_type edge_nodes(2);
  cm->size(esize);
  cm->begin(eiter);
  cm->end(eiter_end);
  FILE *fEdges = fopen(edgesName, "wt");
  if (!fEdges) {
    cerr << "Error opening output file "<<edgesName<<"\n";
    exit(0);
  }
  if (elementsCountHeader) fprintf(fEdges, "%d\n", (unsigned)(esize));
  cerr << "Number of edges = "<< esize <<"\n";
  while(eiter != eiter_end) {
    cm->get_nodes(edge_nodes, *eiter);
    fprintf(fEdges, "%d %d\n", 
	    (int)edge_nodes[0]+baseIndex,
	    (int)edge_nodes[1]+baseIndex);
    ++eiter;
  }
  fclose(fEdges);

  return 0;  
}    
