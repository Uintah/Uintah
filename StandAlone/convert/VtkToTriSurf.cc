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
 *  CVRTItoTriSurfPot.cc
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
  if (argc != 3) {
    cerr << "Usage: "<<argv[0]<<" vtk_poly scirun_field\n";
    return 0;
  }

  FILE *fin = fopen(argv[1], "rt");
  while(fgetc(fin) != '\n');
  while(fgetc(fin) != '\n');
  while(fgetc(fin) != '\n');
  while(fgetc(fin) != '\n');
  int npts;
  fscanf(fin, "POINTS %d float\n", &npts);
  cerr << "Reading "<<npts<<" points from "<<argv[1]<<"...\n";
  int i;
  double x, y, z;
  for (i=0; i<npts; i++) {
    if (fscanf(fin, "%lf %lf %lf", &x, &y, &z) != 3) {
      cerr << "Error - only read "<<npts<<" points!\n";
      exit(0);
    }
    tsm->add_point(Point(x,y,z));
//    cerr << x << " " << y << " " << z <<"\n";
  }
  int nfac, d3;
  char dummy[200];
  fscanf(fin, "%s %d %d\n", dummy, &nfac, &d3);
  cerr << "    and "<<nfac<<" triangles.\n";
  int a, b, c, d;
  for (i=0; i<nfac; i++) {
    if (fscanf(fin, "%d %d %d %d", &a, &b, &c, &d) != 4) {
      cerr << "Error - only read "<<nfac<<" faces!\n";
      exit(0);
    }
    tsm->add_triangle(b, c, d);
//    cerr << b << " " << c << " " << d <<"\n";
  }

  TriSurfMeshHandle tsmH(tsm);
  TriSurfField<double> *ts = scinew TriSurfField<double>(tsmH, Field::NODE);

  FieldHandle tsH(ts);

  TextPiostream out_stream(argv[2], Piostream::Write);
  Pio(out_stream, tsH);
  return 0;  
}    
