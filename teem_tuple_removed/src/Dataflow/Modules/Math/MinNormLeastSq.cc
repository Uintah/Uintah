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
 *  MinNormLeastSq: Select a row or column of a matrix
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 *
 * This module computes the minimal norm, least squared solution to a
 *  nx3 linear system.
 * Given four input ColumnMatrices (v0,v1,v2,b),
 *  find the three coefficients (w0,w1,w2) that minimize:
 *  | (w0v0 + w1v1 + w2v2) - b |.
 * If more than one minimum exisits (the system is under-determined),
 *  choose the coefficients such that (w0,w1,w2) has minimum norm.
 * We output the vector (w0,w1,w2) as a row-matrix,
 *  and we ouput the ColumnMatrix (called x), which is: | w0v0 + w1v1 + w2v2 |.
 *
 */

#include <Core/Math/Mat.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>
#include <math.h>

namespace SCIRun {

class MinNormLeastSq : public Module {
  MatrixIPort* A0_imat_;
  MatrixIPort* A1_imat_;
  MatrixIPort* A2_imat_;
  MatrixIPort* b_imat_;
  MatrixOPort* w_omat_;
  MatrixOPort* bprime_omat_;

public:
  MinNormLeastSq(GuiContext* ctx);
  virtual ~MinNormLeastSq();
  virtual void execute();
};

DECLARE_MAKER(MinNormLeastSq)
MinNormLeastSq::MinNormLeastSq(GuiContext* ctx)
: Module("MinNormLeastSq", ctx, Filter, "Math", "SCIRun")
{
}

MinNormLeastSq::~MinNormLeastSq()
{
}

void
MinNormLeastSq::execute()
{
  A0_imat_ = (MatrixIPort *)get_iport("BasisVec1(Col)");
  A1_imat_ = (MatrixIPort *)get_iport("BasisVec2(Col)");
  A2_imat_ = (MatrixIPort *)get_iport("BasisVec3(Col)");
  b_imat_  = (MatrixIPort *)get_iport("TargetVec(Col)");
  w_omat_  = (MatrixOPort *)get_oport("WeightVec(Col)");
  bprime_omat_  = (MatrixOPort *)get_oport("ResultVec(Col)");
  if (!A0_imat_) {
    error("Unable to initialize iport 'BasisVec1(Col)'.");
    return;
  }
  if (!A1_imat_) {
    error("Unable to initialize iport 'BasisVec2(Col)'.");
    return;
  }
  if (!A2_imat_) {
    error("Unable to initialize iport 'BasisVec3(Col)'.");
    return;
  }
  if (!b_imat_) {
    error("Unable to initialize iport 'TargetVec(Col)'.");
    return;
  }
  if (!w_omat_) {
    error("Unable to initialize oport 'WeightVec(Col)'.");
    return;
  }
  if (!bprime_omat_) {
    error("Unable to initialize oport 'ResultVec(Col)'.");
    return;
  }
  
  int i;
  vector<MatrixHandle> in(4);
  if (!A0_imat_->get(in[0]) || !in[0].get_rep()) { 
    error("No data in BasisVec1"); 
    return;
  }
  if (!A1_imat_->get(in[1]) || !in[1].get_rep()) {
    error("No data in BasisVec2");
    return;
  }
  if (!A2_imat_->get(in[2]) || !in[2].get_rep()) {
    error("No data in BasisVec3");
    return;
  }
  if (!b_imat_->get(in[3]) || !in[3].get_rep()) {
    error("No data in TargetVec");
    return;
  }
  vector<ColumnMatrix *> Ac(4);
  for (i = 0; i < 4; i++) {
    Ac[i] = in[i]->column();
  }
  int size = Ac[0]->nrows();
  for (i = 1; i < 4; i++) {
    if ( Ac[i]->nrows() != size ) {
      error("ColumnMatrices are different sizes");
      return;
    }
  }
  double *A[3];
  for (i=0; i<3; i++) {
    A[i]=Ac[i]->get_data();
  }
  double *b = Ac[3]->get_data();
  double *bprime = new double[size];
  double *x = new double[3];

  min_norm_least_sq_3(A, b, x, bprime, size);
   
  ColumnMatrix* w_vec = new ColumnMatrix(3);
  w_vec->set_data(x);   
  MatrixHandle w_vecH(w_vec);
  w_omat_->send(w_vecH);

  ColumnMatrix* bprime_vec = new ColumnMatrix(size);
  bprime_vec->set_data(bprime);
  MatrixHandle bprime_vecH(bprime_vec);
  bprime_omat_->send(bprime_vecH);
}    

} // End namespace SCIRun
