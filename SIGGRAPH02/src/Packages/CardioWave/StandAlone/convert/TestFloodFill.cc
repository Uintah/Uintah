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
 *  TestFloodFill.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/LatVolField.h>
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
  int minx, miny, minz, maxx, maxy, maxz;
  minx=miny=minz=0;
  maxx=maxy=maxz=32;
  int ni, nj, nk;
  ni=nj=nk=32;
  cout << "Cell centered (1) or node centered (0)? ";
  int c;
  cin >> c;
  
  Point min(minx, miny, minz), max(maxx, maxy, maxz);
  LatVolMeshHandle lvm = new LatVolMesh(ni, nj, nk, min, max);
  FieldHandle fH;
  LatVolField<int> *lv;
  if (c) lv = new LatVolField<int>(lvm, Field::CELL);
  else lv = new LatVolField<int>(lvm, Field::NODE); 
  fH=lv;
  
  for (int i=0; i<ni-1; i++) 
    for (int j=0; j<nj-1; j++)
      for (int k=0; k<nk-1; k++) {
	int val=1;
	if (i==(ni/2) || j==(nj/2) || k==(nk/2)) val=0;
	lv->fdata()(i,j,k)=val;
      }
  TextPiostream out_stream("data.lvi.fld", Piostream::Write);
  Pio(out_stream, fH);
  return 0;  
}    
