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
 *  CardioWaveToLatVolVectorField.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/Vector.h>
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
  if (argc != 7) {
    cerr << "Usage: "<<argv[0]<<" raw nx ny nz samplestride LatVolVectorField\n";
    return 0;
  }
  int ni, nj, nk;
  ni = atoi(argv[2]);
  nj = atoi(argv[3]);
  nk = atoi(argv[4]);
  int sample = atoi(argv[5]);

  FILE *fin = fopen(argv[1], "r");
  if (!fin) {
    cerr << "Error - wasn't able to open file "<<argv[1]<<"\n";
    return 0;
  }
  int n = ni*nj*nk; 
  float *data = (float*)malloc(sizeof(float)*n*3);
  fread(data, sizeof(float)*n*3, 1, fin);
  fclose(fin);

  int x=1;
  if(*(char *)&x == 1) {
    cerr << "Error - data is big endian and this architecture is little endian.\n";
    return 0;
  }

  Point min(0,0,0);

  Point max(1,1,1);
//  Point max(ni/sample, nj/sample, nk/sample);

  LatVolMeshHandle lvm = new LatVolMesh((ni-1)/sample+1, (nj-1)/sample+1, 
					(nk-1)/sample+1, min, max);
  LatVolField<Vector> *lvf = new LatVolField<Vector>(lvm, Field::NODE);
  Vector *vdata = &(lvf->fdata()(0,0,0));
  
  int idx=0;
  int i,j,k;
  for (k=0; k<nk; k++)
    for (j=0; j<nj; j++)
      for (i=0; i<ni; i++, idx+=3)
        if (i%sample==0 && j%sample==0 && k%sample==0)
	  *vdata++ = Vector(data[idx], data[idx+1], data[idx+2]);

  FieldHandle fH(lvf);
  TextPiostream out_stream(argv[6], Piostream::Write);
  Pio(out_stream, fH);
  return 0;
}
