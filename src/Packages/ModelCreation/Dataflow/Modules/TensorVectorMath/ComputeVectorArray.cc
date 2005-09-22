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
 *  ComputeVectorArray.cc:
 *
 *  Written by:
 *   Jeroen Stinstra
 *
 */

#include <Packages/ModelCreation/Core/Algorithms/TVMEngine.h>
#include <Packages/ModelCreation/Core/Algorithms/TVMHelp.h>
#include <Packages/ModelCreation/Core/Algorithms/TVMMath.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>

namespace ModelCreation {

using namespace SCIRun;

class ComputeVectorArray : public Module {
public:
  ComputeVectorArray(GuiContext*);

  virtual ~ComputeVectorArray();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
    GuiString guifunction_;
    GuiString guihelp_;
};


DECLARE_MAKER(ComputeVectorArray)
ComputeVectorArray::ComputeVectorArray(GuiContext* ctx)
  : Module("ComputeVectorArray", ctx, Source, "TensorVectorMath", "ModelCreation"),
  guifunction_(ctx->subVar("function")),
  guihelp_(ctx->subVar("help"))
{
}

ComputeVectorArray::~ComputeVectorArray(){
}

void ComputeVectorArray::execute()
{
  size_t numinputs = (numIPorts()-2);
  TensorVectorMath::TVMArrayList input(numinputs);
  
  if (numinputs > 26)
  {
    error("This module cannot handle more than 26 input matrices");
    return;
  }
  
  char mname = 'A';
  std::string matrixname("A");
  
  int n = 1;


  MatrixIPort *size_iport = dynamic_cast<MatrixIPort *>(getIPort(0));
  MatrixHandle size;
  size_iport->get(size);
  if (size.get_rep())
  {
    if ((size->ncols() != 1)&&(size->nrows()!=1))
    {
      error("Size input needs to be a 1 by 1 matrix");
      return;
    }
    n = static_cast<int>(size->get(0,0));
    if (n == 0) n = 1;
  }
  
  for (size_t p = 0; p < numinputs; p++)
  {

    MatrixIPort *iport = dynamic_cast<MatrixIPort *>(getIPort(p+1));
    MatrixHandle handle;
    iport->get(handle);
    
    if (handle.get_rep())
    {
      if ((handle->ncols()==1)||(handle->ncols()==3)||(handle->ncols()==6)||(handle->ncols()==9))
      {
        matrixname[0] = mname++;
        TensorVectorMath::TVMArray Array(handle,matrixname);
        input[p] = Array;
      }
      else
      {
        std::ostringstream oss;
        oss << "Input matrix " << p+1 << "is not a valid ScalarArray, VectorArray, or TensorArray";
        error(oss.str());
        return;
      }
      
      if (n > 1) 
      {
        if (n != handle->nrows()&&(handle->nrows() != 1))
        {
          std::ostringstream oss;
          oss << "The number of elements in each ScalarArray, VectorArray, or TensorArray is not equal";
          error(oss.str());
          return;          
        }
      }
      else
      {
        n = handle->nrows();
      }
    }
  }
  
  MatrixHandle omatrix = dynamic_cast<Matrix *>(scinew DenseMatrix(n,3));
  if (omatrix.get_rep() == 0)
  {
    error("Could not allocate output matrix");
    return;
  }
  TensorVectorMath::TVMArrayList output(1);
  output[0] = TensorVectorMath::TVMArray(omatrix,"RESULT");
  
  gui->lock();
  gui->eval(getID()+" update_text");
  gui->unlock();
  
  std::string function = guifunction_.get();
  
  TensorVectorMath::TVMEngine engine(this);
  if (!engine.engine(input,output,function,n))
  {
    error("An error occured while executing function");
    return;
  }
  
  MatrixOPort *oport = dynamic_cast<MatrixOPort *>(get_oport(0));
  if (oport)
  {
    oport->send(omatrix);
  }
  
}

void ComputeVectorArray::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("ComputeVectorArray needs a minor command");
    return;
  }

  if( args[1] == "gethelp" )
  {
    TensorVectorMath::TVMHelp Help;
    guihelp_.set(Help.gethelp());
    ctx->reset();
  }

  else
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace ModelCreation


