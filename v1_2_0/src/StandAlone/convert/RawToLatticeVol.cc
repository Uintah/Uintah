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
  if (argc != 3) {
    cerr << "Usage: "<<argv[0]<<" raw lattice\n";
    return 0;
  }

  cout << "What are the dimensions: nx ny nz? ";
  int ni, nj, nk;
  double minx, miny, minz, maxx, maxy, maxz;
  cin >> ni >> nj >> nk;
  cout << "ASCII or Binary (a/b)? ";
  string ascii_or_binary;
  cin >> ascii_or_binary;
  cout << "Datatype (d/f/ui/i/us/s/uc/c)? ";
  string datatype;
  cin >> datatype;
  cout << "Min x y z? ";
  cin >> minx >> miny >> minz;
  cout << "Max x y z? ";
  cin >> maxx >> maxy >> maxz;

  Point min(minx, miny, minz), max(maxx, maxy, maxz);
  LatVolMeshHandle lvm = new LatVolMesh(ni, nj, nk, min, max);
  FieldHandle fH;

  FILE *fin = fopen(argv[1], "rt");

  if (datatype == "d") {
    LatticeVol<double> *lv = 
      new LatticeVol<double>(lvm, Field::NODE);
    fH=lv;
    double *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary == "a") {
      double f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%lf", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(double), ni*nj*nk, fin);
    }
  } else if (datatype == "uc") {
    LatticeVol<unsigned char> *lv = 
      new LatticeVol<unsigned char>(lvm, Field::NODE);
    fH=lv;
    unsigned char *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary == "a") {
      unsigned char f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%c", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(unsigned char), ni*nj*nk, fin);
    }
  } else if (datatype == "f") {
    LatticeVol<float> *lv = 
      new LatticeVol<float>(lvm, Field::NODE);
    fH=lv;
    float *data=&(lv->fdata()(0,0,0));
    if (ascii_or_binary == "a") {
      float f;
      for (int i=0; i<ni*nj*nk; i++) {
	fscanf(fin, "%f", &f);
	*data++ = f;
      }
    } else {
      fread(data, sizeof(float), ni*nj*nk, fin);
    }
  }
  BinaryPiostream out_stream(argv[2], Piostream::Write);
  Pio(out_stream, fH);
  return 0;  
}    
