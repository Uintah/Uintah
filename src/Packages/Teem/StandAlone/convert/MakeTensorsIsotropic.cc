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
 *  MakeTensorsIsotropic.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <teem/ell.h>

#include <Core/Datatypes/TetVolField.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Persistent/Pstreams.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

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
