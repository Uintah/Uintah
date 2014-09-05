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
 *  RawToContourField.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/ContourField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int ptsCountHeader;
int baseIndex;
int edgesCountHeader;
int binOutput;
int debugOn;
void setDefaults() {
  ptsCountHeader=1;
  baseIndex=0;
  edgesCountHeader=1;
  binOutput=0;
  debugOn=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      ptsCountHeader=0;
      currArg++;
    } else if (!strcmp(argv[currArg], "-noEdgesCount")) {
      edgesCountHeader=0;
      currArg++;
    } else if (!strcmp(argv[currArg], "-oneBasedIndexing")) {
      baseIndex=1;
      currArg++;
    } else if (!strcmp(argv[currArg], "-binOutput")) {
      binOutput=1;
      currArg++;
    } else if (!strcmp(argv[currArg], "-debug")) {
      debugOn=1;
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
  ContourMesh *cm = new ContourMesh();
  if (argc < 4 || argc > 7) {
    cerr << "Usage: "<<argv[0]<<" pts edges ContourMesh [-noPtsCount] [-noEdgesCount] [-oneBasedIndexing] [-binOutput] [-debug]\n";
    return 0;
  }
  setDefaults();

  char *ptsName = argv[1];
  char *edgesName = argv[2];
  char *fieldName = argv[3];
  
  if (!parseArgs(argc, argv)) return 0;

  int npts;
  if (!ptsCountHeader) npts = getNumNonEmptyLines(ptsName);
  ifstream ptsstream(ptsName);
  if (ptsCountHeader) ptsstream >> npts;
  cerr << "number of points = "<< npts <<"\n";
  int i;
  for (i=0; i<npts; i++) {
    double x, y, z;
    ptsstream >> x >> y >> z;
    cm->add_node(Point(x,y,z));
    if (debugOn) 
      cerr << "Added point #"<<i<<": ("<<x<<", "<<y<<", "<<z<<")\n";
  }
  cerr << "done adding points.\n";

  int nedges;
  if (!edgesCountHeader) nedges = getNumNonEmptyLines(edgesName);
  ifstream edgesstream(edgesName);
  if (edgesCountHeader) edgesstream >> nedges;
  cerr << "number of edges = "<< nedges <<"\n";
  for (i=0; i<nedges; i++) {
    int n1, n2;
    edgesstream >> n1 >> n2;
    n1-=baseIndex; 
    n2-=baseIndex; 
    if (n1<0 || n1>=npts) { 
      cerr << "Error -- n1 ("<<i<<") out of bounds: "<<n1<<"\n"; 
      return 0; 
    }
    if (n2<0 || n2>=npts) { 
      cerr << "Error -- n2 ("<<i<<") out of bounds: "<<n2<<"\n"; 
      return 0; 
    }
    cm->add_edge(n1, n2);
    if (debugOn) 
      cerr << "Added edge #"<<i<<": ["<<n1<<" "<<n2<<"]\n";
  }
  cerr << "done adding elements.\n";

  cm->flush_changes();
  ContourField<double> *cfd = scinew ContourField<double>(cm, Field::NODE);
  FieldHandle cfdH(cfd);

  if (binOutput) {
    BinaryPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, cfdH);
  } else {
    TextPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, cfdH);
  }
  return 0;  
}    
