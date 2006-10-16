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

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class SetFieldData : public Module {
public:
  SetFieldData(GuiContext*);
  virtual void execute();
private:
  GuiInt keepscalartypegui_;
};


DECLARE_MAKER(SetFieldData)
SetFieldData::SetFieldData(GuiContext* ctx)
  : Module("SetFieldData", ctx, Source, "FieldsData", "ModelCreation"),
  keepscalartypegui_(ctx->subVar("keepscalartype"))
{
}

void SetFieldData::execute()
{
  FieldHandle Input, Output;
  MatrixHandle Data;
  
  if(!(get_input_handle("Field",Input,true))) return;
  if(!(get_input_handle("Data",Data,true))) return;
  
  if (inputs_changed_ || keepscalartypegui_.changed() || !oport_cached("Field"))
  {
    bool keepscalartype = keepscalartypegui_.get();
    SCIRunAlgo::FieldsAlgo algo(this);
    if(!(algo.SetFieldData(Input,Output,Data,keepscalartype))) return;
    
    send_output_handle("Field",Output,false);
  }
}

} // End namespace ModelCreation


