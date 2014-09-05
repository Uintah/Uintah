/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  CollectMatrices: Matrix operations -- concatenate, replace
 *
 *  Written by:
 *   David Weinstein &
 *   Chris Butson
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <stdio.h>

#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class CollectMatrices : public Module {
  MatrixHandle matrixH_;

  GuiInt append_;   // append or replace
  GuiInt row_;      // row or column
  GuiInt front_;    // append at the beginning or the end
  void concat_cols(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out);
  void concat_rows(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out);
public:
  CollectMatrices(GuiContext* ctx);
  virtual ~CollectMatrices();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void *);
};

DECLARE_MAKER(CollectMatrices)
CollectMatrices::CollectMatrices(GuiContext* ctx)
: Module("CollectMatrices", ctx, Filter,"Math", "SCIRun"),
  append_(get_ctx()->subVar("append"), 0),
  row_(get_ctx()->subVar("row"), 0),
  front_(get_ctx()->subVar("front"), 0)
{
}

CollectMatrices::~CollectMatrices()
{
}

void
CollectMatrices::concat_cols(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out) {
    int r, c;
    for (r = 0; r <= m1H->nrows()-1; r++)
    {
      for (c = 0; c <= m1H->ncols()-1; c++)
      {
        out->put(r, c, m1H->get(r,c));
      }
    }

    for (r = 0; r <= m2H->nrows()-1; r++)
    {
      for (c = m1H->ncols(); c <= m1H->ncols()+m2H->ncols()-1; c++)
      {
        out->put(r, c, m2H->get(r,c - m1H->ncols()));
      }
    }
}


void
CollectMatrices::concat_rows(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out) {
    int r, c;
    for (r = 0; r <= m1H->nrows()-1; r++)
    {
      for (c = 0; c <= m1H->ncols()-1; c++)
      {
        out->put(r, c, m1H->get(r,c));
      }
    }

    for (r = m1H->nrows(); r <= m1H->nrows()+m2H->nrows()-1; r++)
    {
      for (c = 0; c <= m2H->ncols()-1; c++)
      {
        out->put(r, c, m2H->get(r - m1H->nrows(), c));
      }
    }
}


void
CollectMatrices::execute()
{
  update_state(NeedData);

  MatrixHandle aH, bH;
  get_input_handle("Optional BaseMatrix", aH, false);
  if (!get_input_handle("SubMatrix", bH)) return;

  DenseMatrix *omatrix = 0;

  bool append = append_.get();
  bool row = row_.get();
  bool front = front_.get();

  if (!append)               // Replace -- just send B matrix
  {
    matrixH_ = bH;
    send_output_handle("CompositeMatrix", matrixH_, true);
    return;
  } 
  else if (!aH.get_rep())    // No A matrix
  { 
    if (!matrixH_.get_rep())
    {
      matrixH_ = bH;
      send_output_handle("CompositeMatrix", matrixH_, true);
      return;
    }
    // No previous CompositeMatrix, so send B
    else {   // Previous CompositeMatrix exists, concatenate with B
      //DenseMatrix *oldmatrix=matrixH_->dense();
      if (row) {
	if (matrixH_->ncols() != bH->ncols()) {
	  warning("SubMatrix and CompositeMatrix must have same number of columns");
	  return;
	} else {
	  omatrix=scinew DenseMatrix(matrixH_->nrows()+bH->nrows(),matrixH_->ncols());
	  if (front) concat_rows(bH,matrixH_,omatrix);
	  else concat_rows(matrixH_,bH,omatrix);
	}
      } else {
	if (matrixH_->nrows() != bH->nrows()) {
	  warning("SubMatrix and CompositeMatrix must have same number of rows");
	  return;
	} else {
	  omatrix=scinew DenseMatrix(matrixH_->nrows(),matrixH_->ncols()+bH->ncols());
	  if (front) concat_cols(bH,matrixH_,omatrix);
	  else concat_cols(matrixH_,bH,omatrix);
	} // columns
      } // rows - columns
    } // previous matrix exists
  } else { // A exists
    if (row) {
      if (aH->ncols() != bH->ncols()) {
	warning("BaseMatrix and CompositeMatrix must have same number of columns");
	return;
      } else {
	omatrix=scinew DenseMatrix(aH->nrows()+bH->nrows(),aH->ncols());
	if (front) concat_rows(bH,aH,omatrix);
	else concat_rows(aH,bH,omatrix);
      }
    } else {
      if (aH->nrows() != bH->nrows()) {
	warning("BaseMatrix and CompositeMatrix must have same number of rows");
	return;
      } else {
	omatrix=scinew DenseMatrix(aH->nrows(),aH->ncols()+bH->ncols());
	if (front) concat_cols(bH,aH,omatrix);
	else concat_cols(aH,bH,omatrix);
      } // columns
    } // rows - columns
  } // A exists

  if (omatrix)
  {
    matrixH_ = omatrix;
    send_output_handle("CompositeMatrix", matrixH_, true);
  }
}


void
CollectMatrices::tcl_command(GuiArgs& args, void* userdata)
{

  if (args[1] == "clear")
  {
    //    DenseMatrix *omatrix = 0;
    matrixH_=0;
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }

}

} // End namespace SCIRun


