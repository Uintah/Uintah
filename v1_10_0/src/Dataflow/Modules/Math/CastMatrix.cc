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

  if (!imat_) {
    error("Unable to initialize iport 'Input'.");
    return;
  }
  if (!omat_) {
    error("Unable to initialize oport 'Output'.");
    return;
  }
  
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
