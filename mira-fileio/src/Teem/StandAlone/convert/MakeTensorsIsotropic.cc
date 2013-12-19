/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  MakeTensorsIsotropic.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 */


#include <teem/ell.h>

#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Persistent/Pstreams.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>

using std::cerr;
using std::ifstream;
using std::endl;

#ifdef _WIN32
#define cbrt(x) pow(x, 1.0/3)
#endif

using namespace SCIRun;

int
main(int argc, char **argv) {
  float a[9];
  ell_3m_to_q_f(a,a);

  FieldHandle handle;
  
  if (argc !=3) {
    cerr << "Usage: "<<argv[0]<<" OldTensorField NewTensorField\n";
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Error - couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading ScalarField from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }

  vector<pair<string, Tensor> > *conds = scinew vector<pair<string, Tensor> >;
  if (!handle->get_property("conductivity_table", *conds)) {
    cerr << "Error - didn't have a conductivity_table in the PropertyManager.\n";
    return 0;
  }
  cerr << "Number of conductivities: "<<conds->size()<<"\n";

  double m[9];
  double eval[3];
  double evec[9];
  int neg_vals=0;
  int small_vals=0;
  for (unsigned int i=0; i<conds->size(); i++) {
    int j, k;
    for (j=0; j<3; j++) for (k=0; k<3; k++) {
      m[j*3+k]=(*conds)[i].second.mat_[j][k];
      (*conds)[i].second.mat_[j][k]=0;
    }
    ell_3m_eigensolve_d(eval, evec, m, 0);
    double cbrt_vol = cbrt(eval[0]*eval[1]*eval[2]);
    if (cbrt_vol<0) { neg_vals++; cbrt_vol=0.1; }
    if (cbrt_vol<0.1) {small_vals++; cbrt_vol=0.1; }
    for (j=0; j<3; j++) (*conds)[i].second.mat_[j][j]=cbrt_vol;
  }
  handle->set_property("conductivity_table", *conds, true);
  cerr << "  # of negative volumes = "<<neg_vals<<"\n";
  cerr << "  # of small volumes = "<<small_vals<<"\n";
  TextPiostream stream2(argv[2], Piostream::Write);
  Pio(stream2, handle);
  return 0;  
}    
