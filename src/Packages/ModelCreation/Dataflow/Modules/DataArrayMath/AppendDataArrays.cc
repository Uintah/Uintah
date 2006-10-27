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

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

namespace ModelCreation {

using namespace SCIRun;

class AppendDataArrays : public Module {
public:
  AppendDataArrays(GuiContext*);
  virtual void execute();
};

DECLARE_MAKER(AppendDataArrays)
AppendDataArrays::AppendDataArrays(GuiContext* ctx)
  : Module("AppendDataArrays", ctx, Source, "DataArrayMath", "ModelCreation")
{
}


void
AppendDataArrays::execute()
{
  std::vector<MatrixHandle> matrixlist;

  if (!(get_dynamic_input_handles("Array",matrixlist,true)));
  
  if (inputs_changed_ || !oport_cached("Array"))
  {
    size_t numinputs = matrixlist.size();
  
    int m = 0;
    int n = 0;
    for (size_t p=0;p<numinputs; p++)
    {
      if (matrixlist[p]->nrows() > 0)
      {
        if (n == 0)
        {
          n = matrixlist[p]->ncols();
        }
        else
        {
          if (n != matrixlist[p]->ncols())
          {
            error("Not every matrix has the same number of columns");
            return;
          }
        }
      }
      m += matrixlist[p]->nrows();
    }
  
    MatrixHandle omatrix = scinew DenseMatrix(m, n);
    double *dataptr = omatrix->get_data_pointer();

    for (size_t p=0;p<numinputs; p++)
    {
      MatrixHandle imatrix = matrixlist[p]->dense();
      double *sdataptr = imatrix->get_data_pointer();
      int ssize = imatrix->nrows()*imatrix->ncols();
      for ( int q = 0; q < ssize; q++)
      {
        *dataptr = *sdataptr;
        dataptr++;
        sdataptr++;
      }
    }
   
    send_output_handle("Array",omatrix,false);
  }
}

} // End namespace ModelCreation


