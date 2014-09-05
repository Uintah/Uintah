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
 *  DecomposeVectorArray.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class DecomposeVectorArray : public Module {
public:
  DecomposeVectorArray(GuiContext*);

  virtual ~DecomposeVectorArray();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(DecomposeVectorArray)
DecomposeVectorArray::DecomposeVectorArray(GuiContext* ctx)
  : Module("DecomposeVectorArray", ctx, Source, "TensorVectorMath", "ModelCreation")
{
}

DecomposeVectorArray::~DecomposeVectorArray(){
}

void DecomposeVectorArray::execute()
{
  MatrixHandle X,Y,Z,V, temp;
  MatrixOPort *oport;
  MatrixIPort *iport;
  
  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("VectorArray"))))
  {
    error("Could not locate input port VectorArray");
    return;
  }
  
  iport->get(V);
  if (V.get_rep() == 0)
  {
    error("The input matrix is empty");
    return;
  }
  
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
  
  temp = dynamic_cast<Matrix *>(V->dense());
  V = temp;
  
  X = dynamic_cast<Matrix *>(scinew DenseMatrix(n,1));
  if (X.get_rep() == 0)
  {
    error("Could allocate memory for output matrix");
    return;
  }
  
  Y = dynamic_cast<Matrix *>(scinew DenseMatrix(n,1));
  if (Y.get_rep() == 0)
  {
    error("Could allocate memory for output matrix");
    return;
  }
  
  Z = dynamic_cast<Matrix *>(scinew DenseMatrix(n,1));
  if (Z.get_rep() == 0)
  {
    error("Could allocate memory for output matrix");
    return;
  }
  
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
  
  if (oport = dynamic_cast<MatrixOPort *>(get_oport("X")))
  {
    oport->send(X);
  }
  
  if (oport = dynamic_cast<MatrixOPort *>(get_oport("Y")))
  {
    oport->send(Y);
  }

  if (oport = dynamic_cast<MatrixOPort *>(get_oport("Z")))
  {
    oport->send(Z);
  }

  
}

void
 DecomposeVectorArray::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


