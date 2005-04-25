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
 *  CVRTItoTetVolFieldPot.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TetVolField.h>
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
  if (argc != 5) {
    cerr << "Usage: "<<argv[0]<<" pts tetras pot fieldout\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream tetstream(argv[2]);
  ifstream potstream(argv[3]);

  int i, npts;
  ptsstream >> npts;
  for (i=0; i<npts; i++) {
    double x, y, z;
    ptsstream >> x;
    if (!ptsstream) { 
      cerr << "Error - only read "<<i<<" of "<<npts<<" points.\n"; 
      return 0; 
    }
    ptsstream >> y >> z;
    tvm->add_point(Point(x,y,z));
  }

  int ntets;
  tetstream >> ntets;
  for (i=0; i<ntets; i++) {
    int j, k, l, m, n, o, p, q, r, s, cond;
    tetstream >> j;
    if (!tetstream) {
      cerr << "Error - only read "<<i<<" of "<<ntets<<" tets.\n";
      return 0;
    }
    tetstream >> k >> l >> m >> n >> o >> p >> q >> r >> s >> cond;

#if 0
    if (i < 10)
      cerr << "j="<<j<<" k="<<k<<" l="<<l<<" m="<<m<<"\n";
    if (i==0) {
      Point pj, pk, pl, pm, pn, po, pp, pq, pr, ps;
      tvm->get_point(pj, j-1);
      tvm->get_point(pk, k-1);
      tvm->get_point(pl, l-1);
      tvm->get_point(pm, m-1);
      tvm->get_point(pn, n-1);
      tvm->get_point(po, o-1);
      tvm->get_point(pp, p-1);
      tvm->get_point(pq, q-1);
      tvm->get_point(pr, r-1);
      tvm->get_point(ps, s-1);
      cerr << "j = "<<pj<<"\n";
      cerr << "k = "<<pk<<"\n";
      cerr << "l = "<<pl<<"\n";
      cerr << "m = "<<pm<<"\n\n";
      cerr << "n = "<<pn<<"\n";
      cerr << "o = "<<po<<"\n";
      cerr << "p = "<<pp<<"\n";
      cerr << "q = "<<pq<<"\n";
      cerr << "r = "<<pr<<"\n";
      cerr << "s = "<<ps<<"\n";
    }
    tvm->add_tet(j-1, k-1, l-1, m-1);
#endif
    tvm->add_tet(j-1, n-1, o-1, p-1);
    tvm->add_tet(k-1, n-1, q-1, s-1);
    tvm->add_tet(l-1, q-1, r-1, o-1);
    tvm->add_tet(m-1, s-1, r-1, p-1);
    tvm->add_tet(q-1, n-1, o-1, s-1);
    tvm->add_tet(q-1, r-1, o-1, s-1);
    tvm->add_tet(p-1, n-1, o-1, s-1);
    tvm->add_tet(p-1, r-1, o-1, s-1);
  }
  
  TetVolMeshHandle tvmH(tvm);
  TetVolField<double> *tv = scinew TetVolField<double>(tvmH, Field::NODE);

  for (i=0; i<npts; i++) {
    double pot;
    if (!potstream) {
      cerr << "Error - only read "<<i<<" of "<<npts<<" potentials.\n"; 
      return 0;
    }
    potstream >> pot;
    tv->fdata()[i]=pot;
  }

  FieldHandle tvH(tv);

  BinaryPiostream out_stream(argv[4], Piostream::Write);
  Pio(out_stream, tvH);
  return 0;  
}    
