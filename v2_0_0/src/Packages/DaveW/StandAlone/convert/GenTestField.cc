/*
 *  GenTestField: Build a test field so we can recognize x/y/z
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/LatVolField.h>
#include <Core/Math/Trig.h>
#include <Core/Persistent/Pstreams.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using std::cerr;
using std::endl;

using namespace SCIRun;

double getVal(int i, int j, int k, int ni, int /*nj*/, int nk) {
  double ii=(Sin(i*1./(ni-1)*10*M_PI)+1./2); // go through 5 periods in i
  double jj=(j%2)/2. +0.5;  // strobe between 1/4 and 3/4 in j
  double kk=k*1./(nk-1);    // ramp from 0-1 for k
  return ii*jj*kk*255.;

}

int main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " outputfile" << endl << "\n";
    return 0;
  }
  int ni = 10;
  int nj = 20;
  int nk = 30;
  LatVolMesh *lvm = scinew LatVolMesh(ni, nj, nk, Point(0,0,0), 
				      Point(ni, nj, nk));
  LatVolField<double> *lv = scinew LatVolField<double>(lvm, Field::NODE);
  FieldHandle fh = lv;
  
  LatVolMesh::Node::iterator niter, niter_end;
  lvm->begin(niter);
  lvm->end(niter_end);
  while (niter != niter_end) {
    lv->fdata()[*niter] = getVal(niter.i_, niter.j_, niter.k_, ni, nj, nk);
    ++niter;
  }
  BinaryPiostream out_stream(argv[1], Piostream::Write);
  Pio(out_stream, fh);
  return 0;
}
