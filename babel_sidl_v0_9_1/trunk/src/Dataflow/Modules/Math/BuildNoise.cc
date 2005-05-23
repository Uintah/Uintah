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
 *  BuildNoise: Add BuildNoise to a matrix
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
#include <Core/Math/Expon.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Math/MusilRNG.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class BuildNoise : public Module {
  MusilRNG musil;
  GuiDouble snr_;
public:
  BuildNoise(GuiContext* ctx);
  virtual ~BuildNoise();
  virtual void execute();
};

DECLARE_MAKER(BuildNoise)
BuildNoise::BuildNoise(GuiContext* ctx)
: Module("BuildNoise", ctx, Filter,"Math", "SCIRun"),
  snr_(ctx->subVar("snr"))
{
}

BuildNoise::~BuildNoise()
{
}

void BuildNoise::execute() {
  MatrixIPort *isignal = (MatrixIPort *)get_iport("Signal");
  MatrixOPort *onoise = (MatrixOPort *)get_oport("Noise");

  update_state(NeedData);
  MatrixHandle matH;
  if (!isignal->get(matH))
    return;
  if (!matH.get_rep()) {
    warning("Empty input matrix.");
    return;
  }

  // gotta make sure we have a Dense or Column matrix...
  // ...if it's Sparse, change it to Dense

  SparseRowMatrix *sm = dynamic_cast<SparseRowMatrix *>(matH.get_rep());
  if (sm) matH = matH->dense();
  else matH.detach();

  double mean, power, sigma;
  mean=power=sigma=0;
  int r, c;
  int nr = matH->nrows();
  int nc = matH->ncols();
  double curr;
  double snr = snr_.get();
  for (r=0; r<nr; r++)
    for (c=0; c<nc; c++) {
      curr = matH->get(r, c);
      mean += curr;
    }
  mean /= nr*nc;
  for (r=0; r<nr; r++)
    for (c=0; c<nc; c++) {
      curr = matH->get(r, c);
      power += (curr-mean)*(curr-mean);
    }
  power /= nr*nc;
  
  sigma = sqrt(power)/(snr*Sqrt(2*M_PI));
  
  for (r=0; r<nr; r++) 
    for (c=0; c<nc; c++) {
      // gaussian distribution about this percentage
      double rnd = 2.0 * musil() - 1.0;
      double perturb = rnd * sigma * sqrt((-2.0 * log(rnd*rnd)) / (rnd*rnd));
      matH->put(r, c, perturb);
    }
  onoise->send(matH);
}
} // End namespace SCIRun
