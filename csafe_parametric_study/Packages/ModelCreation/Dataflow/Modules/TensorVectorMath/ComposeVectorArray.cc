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
 *  ComposeVectorArray.cc:
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

class ComposeVectorArray : public Module {
public:
  ComposeVectorArray(GuiContext*);

  virtual ~ComposeVectorArray();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ComposeVectorArray)
ComposeVectorArray::ComposeVectorArray(GuiContext* ctx)
  : Module("ComposeVectorArray", ctx, Source, "TensorVectorMath", "ModelCreation")
{
}

ComposeVectorArray::~ComposeVectorArray(){
}

void ComposeVectorArray::execute()
{
  MatrixIPort* iport;
  MatrixOPort* oport;
  MatrixHandle X,Y,Z,V;
  
  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("X"))))
  {
    error("Could not locate X input port");
    return;
  }
  iport->get(X);
  if (X.get_rep() == 0)
  {
    error("No input matrix on port X");
    return;
  }

  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("Y"))))
  {
    error("Could not locate Y input port");
    return;
  }
  iport->get(Y);
  if (Y.get_rep() == 0)
  {
    error("No input matrix on port Y");
    return;
  }

  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("Z"))))
  {
    error("Could not locate Z input port");
    return;
  }
  iport->get(Z);
  if (Z.get_rep() == 0)
  {
    error("No input matrix on port Z");
    return;
  }

  int n;
  int Xn, Yn, Zn;
  Xn = X->nrows();
  Yn = Y->nrows();
  Zn = Z->nrows();
  
  n = Xn;
  if (Yn > n) n = Yn;
  if (Zn > n) n = Zn;
  
  if (((Xn!=1)&&(Yn!=1)&&(Xn!=Yn))||((Yn!=1)&&(Zn!=1)&&(Yn!=Zn))||(Xn==0)||(Yn==0)||(Zn==0))
  {
    error("Improper matrix dimensions: all the matrices should have the same number of rows");
    return;
  }

  if (X->ncols()!=1)
  {
    error("X matrix should have only one column");
    return;
  }

  if (Y->ncols()!=1)
  {
    error("Y matrix should have only one column");
    return;
  }

  if (Z->ncols()!=1)
  {
    error("Z matrix should have only one column");
    return;
  }

  MatrixHandle temp;
  temp = dynamic_cast<Matrix *>(X->dense()); 
  X = temp;
  temp = dynamic_cast<Matrix *>(Y->dense());
  Y = temp;
  temp = dynamic_cast<Matrix *>(Z->dense());
  Z = temp;
  
  V = dynamic_cast<Matrix *>(scinew DenseMatrix(n,3));
  
  if (V.get_rep() == 0)
  {
    error("Could not allocate memory for matrix");
    return;
  }
  
  double* xptr = X->get_data_pointer();
  double* yptr = Y->get_data_pointer();
  double* zptr = Z->get_data_pointer();
  double* vptr = V->get_data_pointer();
  
  if ((vptr==0)||(xptr==0)||(yptr==0)||(zptr==0))
  {
    error("Could not allocate enough memory");
    return;
  }
  
  for (int p=0;p<n;p++)
  {
    vptr[0] = *xptr;
    vptr[1] = *yptr;
    vptr[2] = *zptr;
    
    if (Xn > 1) xptr++;
    if (Yn > 1) yptr++;
    if (Zn > 1) zptr++;
    
    vptr += 3;
  }

  if(oport = dynamic_cast<MatrixOPort *>(get_oport("VectorArray")))
  {
    oport->send(V);
  }
}

void
 ComposeVectorArray::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


