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
    int j, k, l, m, cond;
    tetstream >> j;
    if (!tetstream) {
      cerr << "Error - only read "<<i<<" of "<<ntets<<" tets.\n";
      return 0;
    }
    tetstream >> k >> l >> m >> cond;
    tvm->add_tet(j-1, k-1, l-1, m-1);
  }
  
  TetVolMeshHandle tvmH(tvm);
  TetVolField<double> *tv = scinew TetVolField<double>(tvmH, Field::NODE);

  for (i=0; i<npts; i++) {
    double pot;
    if (!potstream) {
      cerr << "Error - only read "<<i<<" of "<<npts<<" potentials.\n"; 
    }
    potstream >> pot;
    tv->fdata()[i]=pot;
  }

  FieldHandle tvH(tv);

  BinaryPiostream out_stream(argv[4], Piostream::Write);
  Pio(out_stream, tvH);
  return 0;  
}    
