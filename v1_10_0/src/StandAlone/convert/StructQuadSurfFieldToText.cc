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
 *  StructQuadSurfFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun StructQuadSurfField, and will save
// out the StructQuadSurfMesh into a .pts file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying ni and nj, unless the user 
// specifies the -noHeader command-line argument.

#include <Core/Datatypes/StructQuadSurfField.h>
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

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" StructQuadSurfField pts [-noHeader]\n\n";
  cerr << "\t This program will read in a SCIRun StructQuadSurfField, and \n";
  cerr << "\t will save out the StructQuadSurfMesh into a .pts file.  The \n";
  cerr << "\t .pts file will specify the x/y/z coordinates of each point, \n";
  cerr << "\t one per line, entries separated by white space; the file \n";
  cerr << "\t will also have a one line header, specifying ni and nj, \n";
  cerr << "\t unless the user specifies the -noHeader command-line \n";
  cerr << "\t argument.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    printUsageInfo(argv[0]);
    return 0;
  }
  setDefaults();

  char *fieldName = argv[1];
  char *ptsName = argv[2];
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
  if (handle->get_type_description(0)->get_name() != "StructQuadSurfField") {
    cerr << "Error -- input field wasn't a StructQuadSurfField (type_name="<<handle->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mH = handle->mesh();
  StructQuadSurfMesh *sqsm = dynamic_cast<StructQuadSurfMesh *>(mH.get_rep());
  StructQuadSurfMesh::Node::iterator niter; 
  StructQuadSurfMesh::Node::iterator niter_end; 
  sqsm->begin(niter);
  sqsm->end(niter_end);
  vector<unsigned int> dims;
  sqsm->get_dim(dims);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    exit(0);
  }
  if (header) fprintf(fPts, "%d %d\n", dims[0], dims[1]);
  cerr << "ni="<< dims[0] << " nj="<<dims[1]<<"\n";
  while(niter != niter_end) {
    Point p;
    sqsm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);
  return 0;  
}    
