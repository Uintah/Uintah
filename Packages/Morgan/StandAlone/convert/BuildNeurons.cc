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
 *  RawToCurveField.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

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
  CurveMesh *cm = new CurveMesh();
  PointCloudMesh *pcm = new PointCloudMesh();
  if (argc != 5) {
    cerr << "Usage: "<<argv[0]<<" nodes connections Contours Dipoles\n";
    return 0;
  }

  char *ptsName = argv[1];
  char *edgesName = argv[2];
  char *contourName = argv[3];
  char *dipoleName = argv[4];
  int npts = getNumNonEmptyLines(ptsName);
  ifstream ptsstream(ptsName);
  cerr << "number of points = "<< npts <<"\n";
  vector<Point> points(npts);
  int i;
  for (i=0; i<npts; i++) {
    int index;
    double x, y, z;
    ptsstream >> index >> x >> y >> z;
    x*=1000000; y*=1000000; z*=1000000;
    cm->add_node(Point(x,y,z));
    points[i]=Point(x,y,z);
  }
  cerr << "done adding points.\n";

  int nedges = getNumNonEmptyLines(edgesName);
  ifstream edgesstream(edgesName);
  cerr << "number of edges = "<< nedges <<"\n";
  vector<Vector> edges(nedges);
  for (i=0; i<nedges; i++) {
    int n1, n2;
    double radius;
    edgesstream >> n1 >> n2 >> radius;
    if (n1<0 || n1>=npts) { 
      cerr << "Error -- n1 ("<<i<<") out of bounds: "<<n1<<"\n"; 
      return 0; 
    }
    if (n2<0 || n2>=npts) { 
      cerr << "Error -- n2 ("<<i<<") out of bounds: "<<n2<<"\n"; 
      return 0; 
    }
    pcm->add_node(points[n1]+(points[n2]-points[n1])/2.);
    cm->add_edge(n1, n2);
    edges[i] = Vector(points[n2]-points[n1]);
  }
  cerr << "done adding edges.\n";

  CurveField<double> *cfd = scinew CurveField<double>(cm, Field::EDGE);
  FieldHandle cfdH(cfd);
  PointCloudField<Vector> *pcv = scinew PointCloudField<Vector>(pcm, Field::NODE);
  for (i=0; i<nedges; i++) pcv->fdata()[i]=edges[i];
  FieldHandle pcvH(pcv);

  TextPiostream out_stream(contourName, Piostream::Write);
  Pio(out_stream, cfdH);
  TextPiostream out_stream2(dipoleName, Piostream::Write);
  Pio(out_stream2, pcvH);

  return 0;  
}    
