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
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {

using namespace SCIRun;

class ReplicateDataArray : public Module {
  public:
    ReplicateDataArray(GuiContext*);
    virtual void execute();
  private:
    GuiInt  guisize_;
};


DECLARE_MAKER(ReplicateDataArray)
ReplicateDataArray::ReplicateDataArray(GuiContext* ctx)
  : Module("ReplicateDataArray", ctx, Source, "DataArrayMath", "SCIRun"),
    guisize_(get_ctx()->subVar("size"))
{
}


void
ReplicateDataArray::execute()
{
  MatrixHandle Input, Size, Output;

  if (!(get_input_handle("DataArray",Input,false))) return;
  get_input_handle("Size",Size,false);

  // this widget has trouble updating properly
  // hence force it to update
  get_gui()->lock();
  get_gui()->eval(get_id()+" update_size");
  get_gui()->unlock();

  if (inputs_changed_ || guisize_.changed() ||
      !oport_cached("Array"))  
  {
  
    int n = 0;  
    if(Size.get_rep() == 0)
    {
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
    
    Output = scinew DenseMatrix(rows*n, cols);
    Input = Input->dense();

    double* outputptr = Output->get_data_pointer();      
    double* inputptr = Input->get_data_pointer(); 
     
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
    
    send_output_handle("Array", Output);
  }
}

} // End namespace SCIRun


