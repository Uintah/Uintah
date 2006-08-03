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

class DomainBoundary : public Module {
public:
  DomainBoundary(GuiContext*);
  virtual void execute();
  
private:
  GuiInt    guiuserange_;
  GuiDouble guiminrange_;
  GuiDouble guimaxrange_;
  GuiInt    guiusevalue_;
  GuiDouble guivalue_;
  GuiInt    guiincludeouterboundary_;
  GuiInt    guiinnerboundaryonly_;
  GuiInt    guinoinnerboundary_;
  GuiInt    guidisconnect_;
  
  
};


DECLARE_MAKER(DomainBoundary)
DomainBoundary::DomainBoundary(GuiContext* ctx)
  : Module("DomainBoundary", ctx, Source, "FieldsCreate", "ModelCreation"),
    guiuserange_(get_ctx()->subVar("userange")),
    guiminrange_(get_ctx()->subVar("minrange")),
    guimaxrange_(get_ctx()->subVar("maxrange")),
    guiusevalue_(get_ctx()->subVar("usevalue")),
    guivalue_(get_ctx()->subVar("value")),
    guiincludeouterboundary_(get_ctx()->subVar("includeouterboundary")),
    guiinnerboundaryonly_(get_ctx()->subVar("innerboundaryonly")),    
    guinoinnerboundary_(get_ctx()->subVar("noinnerboundary")),    
    guidisconnect_(get_ctx()->subVar("disconnect"))    
{
}

void DomainBoundary::execute()
{
  // Define local handles of data objects:
  FieldHandle ifield, ofield;
  MatrixHandle ElemLink;
 
  // Get the new input data: 
  if(!(get_input_handle("Field",ifield,true))) return;
  if (ifield->is_property("ElemLink")) ifield->get_property("ElemLink",ElemLink);
  
  // Only reexecute if the input changed. SCIRun uses simple scheduling
  // that executes every module downstream even if no data has changed:    
  if (inputs_changed_ || guiminrange_.changed() ||  guimaxrange_.changed() ||
      guivalue_.changed() || guiuserange_.changed() || guiincludeouterboundary_.changed() ||
      guiinnerboundaryonly_.changed() || guinoinnerboundary_.changed() || guidisconnect_.changed() ||
      !oport_cached("Field"))
  {
    double minrange, maxrange, value;
    bool   userange, usevalue, includeouterboundary;
    bool   innerboundaryonly, noinnerboundary;
    bool   disconnect;

    // Get all the new input variables:
    minrange = guiminrange_.get();
    maxrange = guimaxrange_.get();
    value    = guivalue_.get();
    userange = static_cast<bool>(guiuserange_.get());
    usevalue = static_cast<bool>(guiusevalue_.get());
    includeouterboundary = static_cast<bool>(guiincludeouterboundary_.get());
    innerboundaryonly = static_cast<bool>(guiinnerboundaryonly_.get());
    noinnerboundary = static_cast<bool>(guinoinnerboundary_.get());
    disconnect = static_cast<bool>(guidisconnect_.get());
    
    // In case we have a one value range, use the range code but for one value:
    if (usevalue) { userange = true; minrange = value; maxrange = value; }
    
    // The innerworks of the module:
    SCIRunAlgo::FieldsAlgo algo(dynamic_cast<ProgressReporter *>(this));
    if(!(algo.DomainBoundary(ifield,ofield,ElemLink,minrange,maxrange,userange,includeouterboundary,innerboundaryonly,noinnerboundary,disconnect))) return;
    
    // send new output if there is any:        
    send_output_handle("Field",ofield,false);
  }
}

} // End namespace ModelCreation


