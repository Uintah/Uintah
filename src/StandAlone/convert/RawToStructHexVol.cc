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
 *  RawToStructHexVol.cc
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/Datatypes/StructHexVolField.h>
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
main(int argc, char **argv)
{
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" nodes conds fieldout\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream condstream(argv[2]);

  int isize, jsize, ksize;
  ptsstream >> isize;
  ptsstream >> jsize;
  ptsstream >> ksize;

  
  StructHexVolMesh *hvm = new StructHexVolMesh(isize, jsize, ksize);

  int i, j, k;
  double x, y, z;
  int count = 0;
  for (i = 0; i < isize; i++)
  {
    for (j = 0; j < jsize; j++)
    {
      for (k = 0; k < ksize; k++)
      {
	ptsstream >> x;
	if (!ptsstream) break;
	ptsstream >> y;
	if (!ptsstream) break;
	ptsstream >> z;
	count++;
	StructHexVolMesh::Node::index_type idx(hvm, i, j, k);
	hvm->set_point(Point(x, y, z), idx);
      }
    }
  }

  if (count != isize * jsize * ksize)
  {
    cerr <<"Error -- was told "<< isize * jsize * ksize <<
      " points, but found " << count << "\n";
    exit(0);
  }
  
  StructHexVolField<int> *hvf =
    scinew StructHexVolField<int>(hvm, Field::CELL);

  count = 0;
  for (i = 0; i < isize-1; i++)
  {
    for (j = 0; j < jsize-1; j++)
    {
      for (k = 0; k < ksize-1; k++)
      {
	int cond;
	if (!ptsstream) break;
	condstream >> cond;
	count++;
	StructHexVolMesh::Cell::index_type idx(hvm, i, j, k);
	hvf->set_value(cond, idx);
      }
    }
  }
  if (count != (isize-1) * (jsize-1) * (ksize-1))
  {
    cerr << "Error -- number of cell centered conditionals does not match the number of points.\n";
    exit(0);
  }

  BinaryPiostream out_stream(argv[3], Piostream::Write);
  FieldHandle hvf_handle(hvf);
  Pio(out_stream, hvf_handle);
  return 0;  
}    
