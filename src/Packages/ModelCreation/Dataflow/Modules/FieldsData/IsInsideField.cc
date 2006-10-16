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

/*
 *  IsInsideField.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>

#include <Core/Algorithms/Fields/FieldsAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class IsInsideField : public Module {
  public:
    IsInsideField(GuiContext*);
    virtual void execute();   
  
  private:
    GuiString outputbasis_;
    GuiString outputtype_;
    GuiDouble outval_;
    GuiDouble inval_;
    GuiInt    partial_inside_;
};


DECLARE_MAKER(IsInsideField)
IsInsideField::IsInsideField(GuiContext* ctx)
  : Module("IsInsideField", ctx, Source, "FieldsData", "ModelCreation"),
    outputbasis_(ctx->subVar("outputbasis")),
    outputtype_(ctx->subVar("outputtype")),
    outval_(ctx->subVar("outval")),
    inval_(ctx->subVar("inval")),
    partial_inside_(ctx->subVar("partial_inside"))
{
}

void IsInsideField::execute()
{
  // Define local handles of data objects:
  FieldHandle input, output;
  FieldHandle object;

  // Get the new input data:  
  if (!(get_input_handle("Field",input,true))) return;
  if (!(get_input_handle("ObjectField",object,true))) return;
  
    // Only reexecute if the input changed. SCIRun uses simple scheduling
  // that executes every module downstream even if no data has changed: 
  if (inputs_changed_ || outputbasis_.changed() || outputtype_.changed() || 
      outval_.changed() || inval_.changed() || partial_inside_.changed() ||
      !oport_cached("Field"))
  {
    SCIRunAlgo::FieldsAlgo algo(this);
    if(!(algo.IsInsideField(input,output,object,outputtype_.get(),outputbasis_.get(),partial_inside_.get(),outval_.get(),inval_.get()))) return;

    // send new output if there is any:    
    send_output_handle("Field",output,false);
  }
}

} // End namespace ModelCreation


