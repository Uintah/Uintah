/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distetbuted under the License is distetbuted on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  TextToContour.cc
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Datatypes/CurveField.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int nodesCountHeader;
int baseIndex;
int edgesCountHeader;
int binOutput;
int debugOn;
void setDefaults() {
  nodesCountHeader=0;
  baseIndex=0;
  edgesCountHeader=0;
  binOutput=0;
  debugOn=0;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      nodesCountHeader=0;
      currArg++;
    } else if (!strcmp(argv[currArg], "-noTetsCount")) {
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
  if (argc < 4 || argc > 7) {
    cerr << "Usage: "<<argv[0]<<" pts segments CurveFieldOut [-noPtsCount] [-noTetsCount] [-oneBasedIndexing] [-binOutput] [-debug]\n";
    return 0;
  }
  setDefaults();

  CurveMesh *mesh = new CurveMesh();

  char *nodes_name = argv[1];
  char *edges_name = argv[2];
  char *field_name = argv[3];
  if (!parseArgs(argc, argv)) return 0;

  int nnodes;
  if (!nodesCountHeader) nnodes = getNumNonEmptyLines(nodes_name);
  ifstream nodes_stream(nodes_name);
  if (nodesCountHeader) nodes_stream >> nnodes;
  if (debugOn) { cout << "Number of nodes = " << nnodes << endl; }
  int i;
  for (i=0; i < nnodes; i++)
  {
    int n;
    double x, y, z;
    nodes_stream >> n >> x >> y >> z;
    Point p(x, y, z);
    mesh->add_node(p);
    if (debugOn)
    {
      cout << "Added point " << i << " at " << p << endl;
    }
  }

  int nedges;
  if (!edgesCountHeader) nedges = getNumNonEmptyLines(edges_name);
  ifstream edges_stream(edges_name);
  if (edgesCountHeader) edges_stream >> nedges;
  if (debugOn) { cout << "Number of edges = " << nedges << endl; }
  vector<double> vals;
  for (i = 0; i < nedges; i++)
  {
    int a, b;
    double val;
    edges_stream >> a >> b >> val;
    mesh->add_edge(a, b);
    vals.push_back(val);
    if (debugOn)
    {
      cout << "Added connection " << i << " from " << a << " to " << b << endl;
    }
  }

  CurveField<double> *field = scinew CurveField<double>(mesh, Field::EDGE);

  CurveMesh::Edge::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);

  while (bi != ei)
  {
    field->set_value(vals[*bi], *bi);
    ++bi;
  }

  FieldHandle field_handle(field);

  if (binOutput) {
    BinaryPiostream out_stream(field_name, Piostream::Write);
    Pio(out_stream, field_handle);
  } else {
    TextPiostream out_stream(field_name, Piostream::Write);
    Pio(out_stream, field_handle);
  }
  return 0;  
}    
