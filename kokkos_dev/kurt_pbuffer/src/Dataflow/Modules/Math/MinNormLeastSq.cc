/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
