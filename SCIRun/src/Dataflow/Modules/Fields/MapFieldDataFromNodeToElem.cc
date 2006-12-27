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

class MapFieldDataFromNodeToElem : public Module {
  public:
    MapFieldDataFromNodeToElem(GuiContext*);
    virtual void execute();

  private:
    GuiString method_;

};


DECLARE_MAKER(MapFieldDataFromNodeToElem)
MapFieldDataFromNodeToElem::MapFieldDataFromNodeToElem(GuiContext* ctx)
  : Module("MapFieldDataFromNodeToElem", ctx, Source, "ChangeFieldData", "SCIRun"),
    method_(get_ctx()->subVar("method"))
{
}

void MapFieldDataFromNodeToElem::execute()
{
  // define input handles:
  FieldHandle input;
  FieldHandle output;
  
  // get data from ports:
  if (!(get_input_handle("Field",input,true))) return;
  
  // Only do work if needed:
  if (inputs_changed_ || method_.changed() || !oport_cached("Field"))
  {
    std::string method = method_.get();
    SCIRunAlgo::FieldsAlgo algo(this);
  
    // Actual algorithm:
    if(!(algo.MapFieldDataFromNodeToElem(input,output,method))) return;
  
    // Send data downstream:
    send_output_handle("Field",output,false);
  }
}

} // End namespace SCIRun

