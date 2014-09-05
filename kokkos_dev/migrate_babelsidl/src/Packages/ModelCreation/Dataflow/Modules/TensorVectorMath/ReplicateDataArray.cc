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
 *  ReplicateDataArray.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class ReplicateDataArray : public Module {
  public:
    ReplicateDataArray(GuiContext*);

    virtual ~ReplicateDataArray();

    virtual void execute();

    virtual void tcl_command(GuiArgs&, void*);
  
  private:
    GuiInt  guisize_;
};


DECLARE_MAKER(ReplicateDataArray)
ReplicateDataArray::ReplicateDataArray(GuiContext* ctx)
  : Module("ReplicateDataArray", ctx, Source, "TensorVectorMath", "ModelCreation"),
    guisize_(get_ctx()->subVar("size"))
{
}

ReplicateDataArray::~ReplicateDataArray(){
}

void
 ReplicateDataArray::execute()
{
  MatrixHandle Input,Size,Output, temp;
  MatrixIPort* iport;
  MatrixOPort* oport;
  
  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("DataArray"))))
  {
    error("Could not locate input port 'Array'");
    return;
  }
  iport->get(Input);
  if(Input.get_rep() == 0)
  {
    error("No matrix was found on input port 'Array'");
    return;
  }
  
  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("Size"))))
  {
    error("Could not locate input port 'Size'");
    return;
  }
  iport->get(Size);
  
  
  int n = 0;  
  if(Size.get_rep() == 0)
  {
    // this widget has trouble updating properly
    // hence force it to update
    get_gui()->lock();
    get_gui()->eval(get_id()+" update_size");
    get_gui()->unlock();
    n = guisize_.get();
  }
  else
  {
    if((Size->ncols() != 1)||(Size->ncols() != 1))
    {
      error("Size needs to be a scalar (1 by 1 matrix)");
      return;
    }
    n = static_cast<int>(Size->get(0,0));
   
    guisize_.set(n);
    get_ctx()->reset(); 
  }

  int rows = 0;
  int cols = 0;
  
  if (n<1)
  {
    error("Size is negative or zero");
    return;
  }

  rows = Input->nrows();
  cols = Input->ncols();

  if ((rows == 0)||(cols == 0))
  {
    error("Array is empty, there is nothing to replicate");
    return;
  }
  
  Output = dynamic_cast<Matrix *>(scinew DenseMatrix(rows*n,cols));
  temp = dynamic_cast<Matrix *>(Input->dense());
  Input = temp;
 
  if ((Output.get_rep() == 0)||(Input.get_rep()==0))
  {
    error("Could not allocate enough memory");
    return;
  }
   
  double* outputptr = Output->get_data_pointer();      
  double* inputptr = Input->get_data_pointer(); 
  if ((inputptr==0)||(outputptr==0))
  {
    error("Could not allocate enough memory");
    return;
  }
   
  outputptr = Output->get_data_pointer(); 
  for (int p =0; p < n; p++)
  {
    inputptr = Input->get_data_pointer();
    for (int q = 0; q < rows*cols; q++)
    {
      *outputptr = *inputptr;
      outputptr++;
      inputptr++;
    }
  }
  
  if (oport = dynamic_cast<MatrixOPort *>(get_oport("Array")))
  {
    oport->send(Output);
  }      
}


void
 ReplicateDataArray::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


