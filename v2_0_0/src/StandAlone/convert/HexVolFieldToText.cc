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
 *  HexVolFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun HexVolField, and will save
// out the HexVolMesh into two files: a .pts file and a .hex file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying number of points, unless the
// user specifies the -noPtsCount command-line argument.
// The .hex file will specify the i/j/k/l/m/n/o/p indices for each hex,
// also one per line, again with a one line header (unless a 
// -noElementsCount flag is used).  The hex entries will be zero-based, 
// unless the user specifies -oneBasedIndexing.

#include <Core/Datatypes/HexVolField.h>
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
  cerr << "\n Usage: "<<progName<<" HexVolField pts hexes [-noPtsCount] [-noElementsCount] [-oneBasedIndexing]\n\n";
  cerr << "\t This program will read in a SCIRun HexVolField, and will \n";
  cerr << "\t save out the HexVolMesh into two files: a .pts file and a \n";
  cerr << "\t .hex file.  The .pts file will specify the x/y/z \n";
  cerr << "\t coordinates of each point, one per line, entries separated \n";
  cerr << "\t by white space; the file will also have a one line header, \n";
  cerr << "\t specifying number of points, unless the user specifies the \n";
  cerr << "\t -noPtsCount command-line argument.  The .hex file will \n";
  cerr << "\t specify the i/j/k/l/m/n/o/p indices for each hex, also one \n";
  cerr << "\t per line, again with a one line header (unless a \n";
  cerr << "\t -noElementsCount flag is used).  The hex entries will be \n";
  cerr << "\t zero-based, unless the user specifies -oneBasedIndexing.\n\n";
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
  char *hexesName = argv[3];
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
  if (handle->get_type_description(0)->get_name() != "HexVolField") {
    cerr << "Error -- input field wasn't a HexVolField (type_name="<<handle->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mH = handle->mesh();
  HexVolMesh *hvm = dynamic_cast<HexVolMesh *>(mH.get_rep());
  HexVolMesh::Node::iterator niter; 
  HexVolMesh::Node::iterator niter_end; 
  HexVolMesh::Node::size_type nsize; 
  hvm->begin(niter);
  hvm->end(niter_end);
  hvm->size(nsize);
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
    hvm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);

  HexVolMesh::Cell::size_type csize; 
  HexVolMesh::Cell::iterator citer; 
  HexVolMesh::Cell::iterator citer_end; 
  HexVolMesh::Node::array_type cell_nodes(8);
  hvm->size(csize);
  hvm->begin(citer);
  hvm->end(citer_end);
  FILE *fHexes = fopen(hexesName, "wt");
  if (!fHexes) {
    cerr << "Error opening output file "<<hexesName<<"\n";
    exit(0);
  }
  size=(unsigned)(csize);
  if (elementsCountHeader) fprintf(fHexes, "%d\n", size);
  cerr << "Number of hexes = "<< csize <<"\n";
  while(citer != citer_end) {
    hvm->get_nodes(cell_nodes, *citer);
    fprintf(fHexes, "%d %d %d %d %d %d %d %d\n", 
	    (int)cell_nodes[0]+baseIndex,
	    (int)cell_nodes[1]+baseIndex,
	    (int)cell_nodes[2]+baseIndex,
	    (int)cell_nodes[3]+baseIndex,
	    (int)cell_nodes[4]+baseIndex,
	    (int)cell_nodes[5]+baseIndex,
	    (int)cell_nodes[6]+baseIndex,
	    (int)cell_nodes[7]+baseIndex);
    ++citer;
  }
  fclose(fHexes);

  return 0;  
}    
