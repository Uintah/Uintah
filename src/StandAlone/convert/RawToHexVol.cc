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
 *  RawToHexVol.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/HexVolField.h>
#include <Core/Containers/Array1.h>
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
  HexVolMesh *hvm = new HexVolMesh();
  if (argc != 5) {
    cerr << "Usage: "<<argv[0]<<" nodes cells conds fieldout\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream cellstream(argv[2]);
  ifstream condstream(argv[3]);

  Array1<HexVolMesh::Node::index_type> nodeIndex;
  int npts;
  ptsstream >> npts;
  int count=0;
  while(ptsstream) {
    double x, y, z;
    ptsstream >> x;
    if (!ptsstream) break;
    ptsstream >> y >> z;
    count++;
    nodeIndex.add(hvm->add_point(Point(x,y,z)));
  }
  if (count != npts) {
    cerr <<"Error -- was told "<<npts<<" points, but found "<<count<<"\n";
    exit(0);
  }

  count=0;
  int ncells;
  cellstream >> ncells;
  while(cellstream) {
    int i0, i1, i2, i3, i4, i5, i6, i7;
    cellstream >> i0;
    if (!cellstream) break;
    cellstream >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7;
    if (i0<0 || i0>=npts) {
      cerr << "Error - i0 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    if (i1<0 || i1>=npts) {
      cerr << "Error - i1 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    if (i2<0 || i2>=npts) {
      cerr << "Error - i2 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    if (i3<0 || i3>=npts) {
      cerr << "Error - i3 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    if (i4<0 || i4>=npts) {
      cerr << "Error - i4 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    if (i5<0 || i5>=npts) {
      cerr << "Error - i5 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    if (i6<0 || i6>=npts) {
      cerr << "Error - i6 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    if (i7<0 || i7>=npts) {
      cerr << "Error - i7 out of bounts (cell "<<count<<")\n";
      exit(0);
    }
    count++;
    hvm->add_hex(nodeIndex[i0], nodeIndex[i1], nodeIndex[i2], nodeIndex[i3],
		 nodeIndex[i4], nodeIndex[i5], nodeIndex[16], nodeIndex[i7]);
  }
  if (count != ncells) {
    cerr <<"Error -- was told "<<ncells<<" cells, but found "<<count<<"\n";
    exit(0);
  }

  Array1<int> conds;
  int nconds;
  condstream >> nconds;

  if (nconds != ncells) {
    cerr << "Error -- ncells is supposed to be equal to nconds, but "<<ncells<<" != "<<nconds<<"\n";
    exit(0);
  }

  while(condstream && (conds.size()<nconds)) {
    int cond;
    condstream >> cond;
    conds.add(cond);
  }
  if (conds.size() != nconds) {
    cerr <<"Error -- was told "<<nconds<<" conductivity entries, but found "<<conds.size()<<"\n";
    exit(0);
  }
  HexVolMeshHandle hvmH(hvm);
  HexVolField<int> *hv = scinew HexVolField<int>(hvmH, Field::CELL);

  for (int ii=0; ii<ncells; ii++)
    hv->fdata()[ii]=conds[ii];
  FieldHandle hvH(hv);

  BinaryPiostream out_stream(argv[4], Piostream::Write);
  Pio(out_stream, hvH);
  return 0;  
}    
