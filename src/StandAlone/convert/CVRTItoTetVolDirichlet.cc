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
 *  CVRTItoTetVolDirichlet.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 2001
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
  if (argc != 6) {
    cerr << "Usage: "<<argv[0]<<" pts tetras pot channels fieldout\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream tetstream(argv[2]);
  ifstream potstream(argv[3]);
  ifstream chanstream(argv[4]);

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
  
  Array1<int> channels;
  char str[100];
  int nlines;
  chanstream >> nlines >> str;
  while(chanstream) {
    int dummy, node_idx;
    chanstream >> dummy;
    if (!chanstream) break;
    chanstream >> node_idx;
    channels.add(node_idx);
  }

  Array1<double> pots;
  while(potstream) {
    double pot;
    potstream >> pot;
    pots.add(pot);
  }

  if (pots.size() != channels.size()) {
    cerr << "Error -- pots and channels files should be the same length (found "<<pots.size()<<" pots in "<<argv[3]<<" and "<< channels.size()<< "channels in "<<argv[4]<<")\n";
    return 0;
  }

  TetVolMeshHandle tvmH(tvm);
  TetVol<double> *tv = scinew TetVol<double>(tvmH, Field::NODE);

  vector<pair<int, double> > dirBC;  
  pair<TetVolMesh::Node::index_type, double> p;
  for (int ii=0; ii<channels.size(); ii++) {
    p.first=channels[ii];
    p.second=pots[ii];
    dirBC.push_back(p);
  }
  
  FieldHandle tvH(tv);
  tvH->store("dirichlet", dirBC);

  TextPiostream out_stream(argv[5], Piostream::Write);
  Pio(out_stream, tvH);
  return 0;  
}    
