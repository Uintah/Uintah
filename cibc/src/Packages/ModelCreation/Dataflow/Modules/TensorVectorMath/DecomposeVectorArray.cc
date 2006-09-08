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

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>

namespace ModelCreation {

using namespace SCIRun;

class DecomposeVectorArray : public Module {
public:
  DecomposeVectorArray(GuiContext*);
  virtual void execute();
};


DECLARE_MAKER(DecomposeVectorArray)
DecomposeVectorArray::DecomposeVectorArray(GuiContext* ctx)
  : Module("DecomposeVectorArray", ctx, Source, "TensorVectorMath", "ModelCreation")
{
}


void
DecomposeVectorArray::execute()
{
  MatrixHandle X,Y,Z,V;
  
  if (!(get_input_handle("VectorArray",V,true)));
  
  if (inputs_changed_ || !oport_cached("X") ||
      !oport_cached("Y") || !oport_cached("Z"))
  {
    int n;
    n = V->nrows();
    
    if (n==0)
    {
      error("The input matrix is empty");
      return;
    }
    
    if (V->ncols() != 3)
    {
      error("Input matrix is not a VectorArray: number of columns is not 3");
      return;
    }
    
    V = V->dense();
    X = scinew DenseMatrix(n, 1);
    Y = scinew DenseMatrix(n, 1);
    Z = scinew DenseMatrix(n, 1);
    
    double* vptr = V->get_data_pointer();
    double* xptr = X->get_data_pointer();
    double* yptr = Y->get_data_pointer();
    double* zptr = Z->get_data_pointer();
    
    if ((vptr==0)||(xptr==0)||(yptr==0)||(zptr==0))
    {
      error("Could not allocate enough memory");
      return;
    }
    
    for (int p=0; p<n ;p++)
    {
      *xptr = vptr[0];
      *yptr = vptr[1];
      *zptr = vptr[2];
      
      vptr += 3;
      xptr++;
      yptr++;
      zptr++;
    }
    
    send_output_handle("X",X,false);
    send_output_handle("Y",Y,false);
    send_output_handle("Z",Z,false);
  }
}

} // End namespace CardioWave

