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
 *  Submatrix: Clip out a subregion from a Matrix
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {

class Submatrix : public Module
{
private:
  GuiString mincol_;
  GuiString maxcol_;
  GuiString minrow_;
  GuiString maxrow_;
  GuiInt nrow_;
  GuiInt ncol_;

public:
  Submatrix(GuiContext* ctx);
  virtual ~Submatrix();

  virtual void execute();
};


DECLARE_MAKER(Submatrix)

Submatrix::Submatrix(GuiContext* ctx)
  : Module("Submatrix", ctx, Filter,"Math", "SCIRun"),
    mincol_(ctx->subVar("mincol")),
    maxcol_(ctx->subVar("maxcol")),
    minrow_(ctx->subVar("minrow")),
    maxrow_(ctx->subVar("maxrow")),
    nrow_(ctx->subVar("nrow")),
    ncol_(ctx->subVar("ncol"))
{
}


Submatrix::~Submatrix()
{
}


void
Submatrix::execute()
{
  MatrixIPort *imp = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrix;
  if (!(imp && imp->get(imatrix) && imatrix.get_rep()))
  {
    return;
  }
  nrow_.set(imatrix->nrows());
  ncol_.set(imatrix->ncols());
  
  MatrixOPort *omp = (MatrixOPort *)get_oport("Output Matrix");
  if (!omp)
  {
    error("Could not open output matrix port.");
    return;
  }

  MatrixIPort *cmp = (MatrixIPort *)get_iport("Optional Range Bounds");
  MatrixHandle cmatrix;
  int mincol, maxcol, minrow, maxrow;
  if (cmp && cmp->get(cmatrix) && cmatrix.get_rep())
  {
    // Grab the bounds from the clip matrix, check them, and update the gui.
    if (cmatrix->nrows() > 1)
    {
      minrow = (int)cmatrix->get(0, 0);
      maxrow = (int)cmatrix->get(1, 0);

      if (cmatrix->ncols() > 1)
      {
	mincol = (int)cmatrix->get(0, 1);
	maxcol = (int)cmatrix->get(1, 1);
      }
      else
      {
	mincol = 0;
	maxcol = imatrix->ncols() - 1;
      }
    }
    else
    {
      minrow = 0;
      maxrow = imatrix->ncols() - 1;
      mincol = 0;
      maxcol = imatrix->ncols() - 1;
    }
  }
  else
  {
    if (!string_to_int(minrow_.get(), minrow)) minrow = 0;
    if (!string_to_int(maxrow_.get(), maxrow)) maxrow = imatrix->nrows()-1;
    if (!string_to_int(mincol_.get(), mincol)) mincol = 0;
    if (!string_to_int(maxcol_.get(), maxcol)) maxcol = imatrix->ncols()-1;
  }

  minrow = Min(Max(0, minrow), imatrix->nrows()-1);
  maxrow = Min(Max(0, maxrow), imatrix->nrows()-1);
  mincol = Min(Max(0, mincol), imatrix->ncols()-1);
  maxcol = Min(Max(0, maxcol), imatrix->ncols()-1);

  minrow_.set(to_string(minrow));
  maxrow_.set(to_string(maxrow));
  mincol_.set(to_string(mincol));
  maxcol_.set(to_string(maxcol));
    
  if (mincol > maxcol || minrow > maxrow)
  {
    warning("Max range must be greater than or equal to min range, disregarding.");
    return;
  }

  // No need to clip if the matrices are identical.
  if (minrow == 0 && maxrow == (imatrix->nrows()-1) &&
      mincol == 0 && maxcol == (imatrix->ncols()-1))
  {
    omp->send(imatrix);
  }
  else
  {
    MatrixHandle omatrix = imatrix->submatrix(minrow, mincol, maxrow, maxcol);
    omp->send(omatrix);
  }
}


} // End namespace SCIRun
