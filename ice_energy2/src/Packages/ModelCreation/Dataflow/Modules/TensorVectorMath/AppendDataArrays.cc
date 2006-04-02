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
 *  AppendDataArrays.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>


#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class AppendDataArrays : public Module {
public:
  AppendDataArrays(GuiContext*);
  virtual void execute();
};

DECLARE_MAKER(AppendDataArrays)
AppendDataArrays::AppendDataArrays(GuiContext* ctx)
  : Module("AppendDataArrays", ctx, Source, "TensorVectorMath", "ModelCreation")
{
}

void AppendDataArrays::execute()
{
  MatrixIPort *iport;
  MatrixOPort *oport;
  
  size_t numinputs = (num_input_ports()-1);
  
  std::vector<MatrixHandle> matrixlist;
  int n, m;
  
  m = 0;
  n = 0;
  for (size_t p=0;p<numinputs; p++)
  {
    if(!(iport = dynamic_cast<MatrixIPort *>(get_input_port(p))))
    {
      error("Could not locate matrix input port");
      return;
    }
    
    iport->get(matrixlist[p]);
    
    if (matrixlist[p].get_rep() == 0)
    {
      error("One of the input ports has no matrix");
      return;
    }
    
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
  
  MatrixHandle omatrix = dynamic_cast<Matrix *>(scinew DenseMatrix(m,n));
  
  if (omatrix.get_rep() == 0)
  {
    error("Could not allocate destination matrix");
    return;
  }
  
  if (omatrix->get_data_pointer() == 0)
  {
    error("Could not allocate destination matrix");
    return;
  }

  double *dataptr = omatrix->get_data_pointer();

  for (size_t p=0;p<numinputs; p++)
  {
    MatrixHandle imatrix = dynamic_cast<Matrix *>(matrixlist[p]->dense());
    if (imatrix.get_rep() == 0)
    {
      error("Could not convert matrix into dense matrix");
      return;
    }
    double *sdataptr = imatrix->get_data_pointer();
    int ssize = imatrix->nrows()*imatrix->ncols();
    for ( size_t q = 0; q < ssize; q++)
    {
      *dataptr = *sdataptr;
      dataptr++;
      sdataptr++;
    }
  }
  
  if (oport = dynamic_cast<MatrixOPort *>(get_output_port(0)))
  {
    oport->send(omatrix);
  }
}

} // End namespace ModelCreation


