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
 *  CVRTItoTriSurfFieldPot.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TriSurfField.h>
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
  TriSurfMesh *tsm = new TriSurfMesh();
  if (argc != 5 && argc != 6) {
    cerr << "Usage: "<<argv[0]<<" pts fac pot [channels] fieldout\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);
  ifstream facstream(argv[2]);
  ifstream potstream(argv[3]);

  int npts=0;
  while(ptsstream) {
    double x, y, z;
    ptsstream >> x;
    if (!ptsstream) break;
    ptsstream >> y >> z;
    npts++;
    tsm->add_point(Point(x,y,z));
  }

  int nfacs=0;
  while(facstream) {
    int i, j, k;
    facstream >> i;
    if (!facstream) break;
    facstream >> j >> k;
    nfacs++;
    tsm->add_triangle(i-1, j-1, k-1);
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

  Array1<double> pots;
  while(potstream) {
    double pot;
    potstream >> pot;
    pots.add(pot);
  }

  TriSurfMeshHandle tsmH(tsm);
  TriSurfField<double> *ts = scinew TriSurfField<double>(tsmH, Field::NODE);

  for (ii=0; ii<npts; ii++)
    ts->fdata()[ii]=pots[channels[ii]];
  FieldHandle tsH(ts);

  if (argc == 6) {
    TextPiostream out_stream(argv[5], Piostream::Write);
    Pio(out_stream, tsH);
  } else {
    TextPiostream out_stream(argv[4], Piostream::Write);
    Pio(out_stream, tsH);
  }    
  return 0;  
}    
