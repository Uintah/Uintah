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
 *  RawToLatticeVol.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/LatticeVol.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;
using std::cin;
using std::cout;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" raw lattice\n";
    return 0;
  }

  ifstream rawstream(argv[1]);

  cout << "What are the dimensions: nx ny nz? ";
  int ni, nj, nk;
  cin >> ni >> nj >> nk;
  cout << "ASCII or Binary (a/b)? ";
  string ascii_or_binary;
  cin >> ascii_or_binary;
  cout << "Datatype (d/f/ui/i/us/s/uc/c)? ";
  string datatype;
  cin >> datatype;

  cerr << "\n\n ni="<<ni<<" nj="<<nj<<" nk="<<nk<<"\n";
  cerr << "ascii_or_binary="<<ascii_or_binary<<"\n";
  cerr << "Datatype="<<datatype<<"\n";

  Point min(0,0,0), max(ni,nj,nk);
  LatVolMeshHandle lvm = new LatVolMesh(ni, nj, nk, min, max);
  LatticeVol<double> *lv = new LatticeVol<double>(lvm, Field::NODE);
  FieldHandle fH(lv);

  TextPiostream out_stream(argv[3], Piostream::Write);
  Pio(out_stream, fH);
  return 0;  
}    
