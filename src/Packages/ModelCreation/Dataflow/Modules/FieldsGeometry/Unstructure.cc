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

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class Unstructure : public Module {
public:
  Unstructure(GuiContext*);

  virtual void execute();
};


DECLARE_MAKER(Unstructure)
Unstructure::Unstructure(GuiContext* ctx)
  : Module("Unstructure", ctx, Source, "FieldsGeometry", "ModelCreation")
{
}

void Unstructure::execute()
{
  FieldHandle ifield, ofield;
  
  if (!(get_input_handle("Field",ifield,true))) return;
  
  if (inputs_changed_ || !oport_cached("Field"))
  {
    SCIRunAlgo::FieldsAlgo algo(dynamic_cast<ProgressReporter *>(this));
    if (!(algo.Unstructure(ifield,ofield))) return;
    
    send_output_handle("Field",ofield,false);
  }
}

} // End namespace ModelCreation


