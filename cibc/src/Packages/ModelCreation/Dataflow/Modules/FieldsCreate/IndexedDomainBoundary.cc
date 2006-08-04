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
#include <Core/Malloc/Allocator.h>

#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>

namespace ModelCreation {

using namespace SCIRun;

class IndexedDomainBoundary : public Module {
public:
  IndexedDomainBoundary(GuiContext*);
  virtual void execute();
  
};


DECLARE_MAKER(IndexedDomainBoundary)
IndexedDomainBoundary::IndexedDomainBoundary(GuiContext* ctx)
  : Module("IndexedDomainBoundary", ctx, Source, "FieldsCreate", "ModelCreation")
{
}

void IndexedDomainBoundary::execute()
{
  // Define local handles of data objects:
  FieldHandle ifield, ofield;
  MatrixHandle ElemLink;
  
  // Get the new input data:   
  if(!(get_input_handle("Field",ifield,true))) return;
  if (ifield->is_property("ElemLink")) ifield->get_property("ElemLink",ElemLink);

  // Only reexecute if the input changed. SCIRun uses simple scheduling
  // that executes every module downstream even if no data has changed: 
  if (inputs_changed_ || !oport_cached("Field"))
  {  
    SCIRunAlgo::FieldsAlgo algo(dynamic_cast<ProgressReporter *>(this));  
    if(!(algo.IndexedDomainBoundary(ifield,ofield,ElemLink,0.0,0.0,false,true,false,false,false))) return;

    // send new output if there is any:        
    send_output_handle("Field",ofield,false);
  }
}

} // End namespace ModelCreation

