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

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>
#include <Dataflow/Network/Module.h>

namespace ModelCreation {

using namespace SCIRun;

class FindClosestNodeIndexByValue : public Module {
public:
  FindClosestNodeIndexByValue(GuiContext*);
  virtual void execute();
};


DECLARE_MAKER(FindClosestNodeIndexByValue)
FindClosestNodeIndexByValue::FindClosestNodeIndexByValue(GuiContext* ctx)
  : Module("FindClosestNodeIndexByValue", ctx, Source, "FieldsGeometry", "ModelCreation")
{
}

void FindClosestNodeIndexByValue::execute()
{
  FieldHandle field, points;
  MatrixHandle value;
  
  if (!(get_input_handle("Field",field,true))) return;
  if (!(get_input_handle("Points",points,true))) return;
  if (!(get_input_handle("Value",value,true))) return;
  
  if (inputs_changed_ || !oport_cached("Indices"))
  {
    MatrixHandle indices;
    double val;
    
    SCIRunAlgo::FieldsAlgo algo(this);
    SCIRunAlgo::ConverterAlgo calgo(this);
    
    if (!(calgo.MatrixToDouble(value,val))) return;
    if (!(algo.FindClosestNodeByValue(field,indices,points,val))) return;
    
    send_output_handle("Indices",indices,false);
  }
}

} // End namespace ModelCreation


