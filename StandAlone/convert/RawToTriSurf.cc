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

#include <Core/Datatypes/TriSurf.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Datatypes/FieldSet.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  TriSurfMesh *tsm = new TriSurfMesh();
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" pts tris TriSurf\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream tristream(argv[2]);

  int i;
  ptsstream >> i;
  cerr << "number of points = "<<i<<"\n";
  for (; i>0; i--) {
    double x, y, z;
    ptsstream >> x >> y >> z;
    tsm->add_point(Point(x,y,z));
  }
  cerr << "done adding points.\n";

  while (tristream) {
    int n1, n2, n3;
    tristream >> n1;
    if (!tristream) break;
    tristream >> n2 >> n3;
    tsm->add_triangle(n1, n2, n3);
  }
  cerr << "done adding elements.\n";

  tsm->connect();
  TriSurfMeshHandle tsmh(tsm);

  TextPiostream out_stream(argv[3], Piostream::Write);
  Pio(out_stream, tsmh);
  return 0;  
}    
