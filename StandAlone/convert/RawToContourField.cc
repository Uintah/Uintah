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

#include <Core/Datatypes/ContourMesh.h>
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
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" pts edges TriSurf\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream edgesstream(argv[2]);
  char idx[100];
  double x, y, z;
  while (ptsstream) {
    ptsstream >> idx >> x >> y >> z;
    cm->add_node(Point(x,y,z));
  }

  while (edgesstream) {
    int n1, n2;
    double rad;
    edgesstream >> n1;
    if (!edgesstream) break;
    edgesstream >> n2 >> rad;
    cm->add_edge(n1, n2);
  }

  ContourMeshHandle cmh(cm);

  TextPiostream out_stream(argv[3], Piostream::Write);
  Pio(out_stream, cmh);
  return 0;  
}    
