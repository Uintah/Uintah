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
  if (argc != 4 && argc != 5) {
    cerr << "Usage: "<<argv[0]<<" pts tris [vals] TriSurf\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream tristream(argv[2]);

  int i, npts;
  ptsstream >> npts;
  cerr << "number of points = "<<npts<<"\n";
  for (i=0; i<npts; i++) {
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

  TriSurf<double> *tsd = scinew TriSurf<double>(tsm, Field::NODE);
  FieldHandle fh(tsd);

  if (argc == 5) {
    ifstream valstream(argv[3]);
    for (i=0; i<npts; i++) {
      double val;
      valstream >> val;
      tsd->fdata()[i]=val;
    }
    cerr << "done adding values.\n";
  }

  if (argc == 4) {
    TextPiostream out_stream(argv[3], Piostream::Write);
    Pio(out_stream, fh);
  } else {
    TextPiostream out_stream(argv[4], Piostream::Write);
    Pio(out_stream, fh);
  }

  return 0;  
}    
