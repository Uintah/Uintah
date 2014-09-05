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
 *  CVRTItoTetVolGrad.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TetVol.h>
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
  TetVolMesh *tvm = new TetVolMesh();
  if (argc != 5 && argc != 6) {
    cerr << "Usage: "<<argv[0]<<" pts tetras grad [channels] fieldout\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream tetstream(argv[2]);
  ifstream gradstream(argv[3]);

  int npts=0;
  while(ptsstream) {
    double x, y, z;
    ptsstream >> x;
    if (!ptsstream) break;
    ptsstream >> y >> z;
    npts++;
    tvm->add_point(Point(x,y,z));
  }

  int ntets=0;
  while(tetstream) {
    int i, j, k, l;
    tetstream >> i;
    if (!tetstream) break;
    tetstream >> j >> k >> l;
    ntets++;
    tvm->add_tet(i-1, j-1, k-1, l-1);
  }
  
  Array1<int> channels(npts);
  int ii;
  for (ii=0; ii<npts; ii++) channels[ii]=ii;

  if (argc == 6) {
    ifstream chanstream(argv[4]);
    char str[100];
    int nlines;
    chanstream >> nlines >> str;
    while(chanstream) {
      int oldnode, newnode;
      chanstream >> oldnode;
      if (!chanstream) break;
      chanstream >> newnode;
      channels[oldnode-1]=newnode-1;
    }
  }

  Array1<Vector> grads;
  while(gradstream) {
    double sx, sy, sz, ex, ey, ez;
    gradstream >> sx >> sy >> sz;
    if (!gradstream) break;
    gradstream >> ex >> ey >> ez;
    grads.add(Point(ex, ey, ez)-Point(sx, sy, sz));
  }

  TetVolMeshHandle tvmH(tvm);
  TetVol<Vector> *tv = scinew TetVol<Vector>(tvmH, Field::NODE);

  for (ii=0; ii<npts; ii++)
    tv->fdata()[ii]=grads[channels[ii]];
  FieldHandle tvH(tv);

  if (argc == 6) {
    TextPiostream out_stream(argv[5], Piostream::Write);
    Pio(out_stream, tvH);
  } else {
    TextPiostream out_stream(argv[4], Piostream::Write);
    Pio(out_stream, tvH);
  }    
  return 0;  
}    
