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
 *  FlatToTriSurf.cc
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
#include <Core/Datatypes/PointCloud.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  ContourMesh *cm = new ContourMesh();
  if (argc != 5) {
    cerr << "Usage: "<<argv[0]<<" pts edges ContourField [PointCloud]\n";
    return 0;
  }

  PointCloudMesh *pm = new PointCloudMesh();

  ifstream ptsstream(argv[1]);
  ifstream edgesstream(argv[2]);
  Array1<Point> pts;
  Point p;
  Array1<Vector> vecs;
  Vector v;
  char idx[100];
  double x, y, z;
  while (ptsstream) {
    ptsstream >> idx >> x >> y;
    if (!ptsstream) break;
    ptsstream >> z;
    p = Point(x,y,z);
    pts.add(p);
    cm->add_node(p);
  }
  
  while (edgesstream) {
    int n1, n2;
    double rad;
    edgesstream >> n1;
    if (!edgesstream) break;
    edgesstream >> n2 >> rad;
    cm->add_edge(n1,n2);
    v = pts[n2]-pts[n1];
    pm->add_node(pts[n1]+v/2.);
    v.normalize();
    vecs.add(v);
  }

  FieldHandle fh;
  ContourField<Vector> *cf = scinew ContourField<Vector>(cm, Field::EDGE);
  int i;
  for (i=0; i<vecs.size(); i++)
    cf->fdata()[i] = vecs[i];
  TextPiostream out_stream(argv[3], Piostream::Write);
  fh = cf;
  Pio(out_stream, fh);

  PointCloud<Vector> *pc = scinew PointCloud<Vector>(pm, Field::NODE);
  for (i=0; i<vecs.size(); i++)
    pc->fdata()[i] = vecs[i];
  TextPiostream out_stream2(argv[4], Piostream::Write);
  fh = pc;

  if (argc == 5) 
    Pio(out_stream2, fh);
  
  return 0;  
}    
