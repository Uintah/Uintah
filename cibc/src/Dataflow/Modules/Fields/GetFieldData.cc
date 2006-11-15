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

namespace SCIRun {

class GetFieldData : public Module {
public:
  GetFieldData(GuiContext*);
  virtual void execute();
};


DECLARE_MAKER(GetFieldData)
GetFieldData::GetFieldData(GuiContext* ctx)
  : Module("GetFieldData", ctx, Source, "ChangeFieldData", "SCIRun")
{
}

void GetFieldData::execute()
{
  // define input handles:
  FieldHandle Input;
  MatrixHandle Data;
  
  // Get data from ports:
  if(!(get_input_handle("Field",Input,true))) return;
  
  // Only do work if inputs changed:
  if (inputs_changed_ || !oport_cached("Data"))
  {
    SCIRunAlgo::FieldsAlgo algo(this);
    
    // Actual algorithm:
    if(!(algo.GetFieldData(Input,Data))) return;
  
    // Send output downstream:
    send_output_handle("Data",Data,false);
  }
}

} // End namespace SCIRun

