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
 *  CastMatrix: Unary matrix operations -- transpose, negate
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class CastMatrix : public Module {
  MatrixIPort* imat_;
  MatrixOPort* omat_;

  GuiString oldtype_;
  GuiString newtype_;
  GuiInt nrow_;
  GuiInt ncol_;
public:
  CastMatrix(GuiContext* ctx);
  virtual ~CastMatrix();
  virtual void execute();
};

DECLARE_MAKER(CastMatrix)
CastMatrix::CastMatrix(GuiContext* ctx)
: Module("CastMatrix", ctx, Filter,"Math", "SCIRun"),
  oldtype_(ctx->subVar("oldtype")), newtype_(ctx->subVar("newtype")),
  nrow_(ctx->subVar("nrow")), ncol_(ctx->subVar("ncol"))
{
}

CastMatrix::~CastMatrix()
{
}

void CastMatrix::execute() {
  imat_ = (MatrixIPort *)get_iport("Input");
  omat_ = (MatrixOPort *)get_oport("Output");
  
  update_state(NeedData);
  MatrixHandle imH;
  if (!imat_->get(imH))
    return;
  if (!imH.get_rep()) {
    warning("Empty input matrix.");
    return;
  }
  nrow_.set(imH->nrows());
  ncol_.set(imH->ncols());

  oldtype_.set(imH->type_name());

  string newtype = newtype_.get();
  MatrixHandle omH;

  if (newtype == "DenseMatrix") {
    omH = imH->dense();
  } else if (newtype == "SparseRowMatrix") {
    omH = imH->sparse();
  } else if (newtype == "ColumnMatrix") {
    omH = imH->column();
  } else if (newtype == "Same") {
    omH = imH;
  } else {
    error("CastMatrix: unknown cast type "+newtype);
    return;
  }
  omat_->send(omH);
}
} // End namespace SCIRun
