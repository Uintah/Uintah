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
#include <iostream>
#include <sstream>

namespace SCIRun {

class Submatrix : public Module
{
private:
  GuiInt mincol_;
  GuiInt maxcol_;
  GuiInt minrow_;
  GuiInt maxrow_;

public:
  Submatrix(const string& id);
  virtual ~Submatrix();

  virtual void execute();
};


extern "C" Module* make_Submatrix(const string& id)
{
    return new Submatrix(id);
}


Submatrix::Submatrix(const string& id)
  : Module("Submatrix", id, Filter,"Math", "SCIRun"),
    mincol_("mincol", id, this),
    maxcol_("maxcol", id, this),
    minrow_("minrow", id, this),
    maxrow_("maxrow", id, this)
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

    mincol = Max(0, mincol);
    minrow = Max(0, minrow);
    maxcol = Min(maxcol + 1, imatrix->ncols());
    maxrow = Min(maxrow + 1, imatrix->nrows());

    mincol_.set(mincol);
    maxcol_.set(maxcol);
    minrow_.set(minrow);
    maxrow_.set(maxrow);
  }
  else
  {
    // Get the bounds, clip to the input matrix size.
    if (mincol_.get() < 0) { mincol_.set(0); }
    if (minrow_.get() < 0) { minrow_.set(0); }
    if (maxcol_.get() < 0) { maxcol_.set(imatrix->ncols() - 1); }
    if (maxrow_.get() < 0) { maxrow_.set(imatrix->nrows() - 1); }

    mincol = mincol_.get();
    maxcol = maxcol_.get();
    minrow = minrow_.get();
    maxrow = maxrow_.get();

    mincol = Max(0, mincol);
    minrow = Max(0, minrow);
    maxcol = Min(maxcol + 1, imatrix->ncols());
    maxrow = Min(maxrow + 1, imatrix->nrows());
  }
    
  if (mincol >= maxcol || minrow >= maxrow)
  {
    warning("Zero size output matrix, disregarding.");
    return;
  }

  // No need to clip if the matrices are identical.
  if (minrow == 0 && maxrow == imatrix->nrows() &&
      mincol == 0 && maxcol == imatrix->ncols())
  {
    omp->send(imatrix);
    return;
  }

  // Build a dense matrix with the clipped values in it.
  DenseMatrix *omatrix = scinew DenseMatrix(maxrow - minrow, maxcol - mincol);
  int r, c;
  for (r = minrow; r < maxrow; r++)
  {
    for (c = mincol; c < maxcol; c++)
    {
      omatrix->put(r, c, imatrix->get(r + minrow, c + mincol));
    }
  }
  
  omp->send(omatrix);
}


} // End namespace SCIRun
