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
 *  CVRTItoPointCloudFieldPot.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/PointCloudField.h>
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
  PointCloudMesh *pcm = new PointCloudMesh();
  if (argc != 3 && argc != 4) {
    cerr << "Usage: "<<argv[0]<<" pts [pot] fieldout\n";
    return 0;
  }

  ifstream ptsstream(argv[1]);

  int npts=0;
  while(ptsstream) {
    double x, y, z;
    ptsstream >> x;
    if (!ptsstream) break;
    ptsstream >> y >> z;
    npts++;
    pcm->add_point(Point(x,y,z));
  }

  PointCloudMeshHandle pcmH(pcm);
  PointCloudField<double> *pc = scinew PointCloudField<double>(pcmH, Field::NODE);

  int ii;
  for (ii=0; ii<npts; ii++)
    pc->fdata()[ii]=0;

  if (argc == 4) {
    ifstream potstream(argv[2]);
    for (ii=0; ii<npts; ii++) {
      double pot;
      potstream >> pot;
      pc->fdata()[ii] = pot;
    }
  }

  FieldHandle pcH(pc);

  if (argc == 3) {
    TextPiostream out_stream(argv[2], Piostream::Write);
    Pio(out_stream, pcH);
    return 0;  
  } else {
    TextPiostream out_stream(argv[3], Piostream::Write);
    Pio(out_stream, pcH);
    return 0;  
  }
}    
