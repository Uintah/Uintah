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

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>


namespace ModelCreation {

using namespace SCIRun;

class ModalMapping : public Module {
  public:
    ModalMapping(GuiContext*);
    virtual void execute();

  private:
    GuiString mappingmethod_;
    GuiString integrationmethod_;
    GuiString integrationfilter_;
    GuiDouble def_value_;
};


DECLARE_MAKER(ModalMapping)
ModalMapping::ModalMapping(GuiContext* ctx)
  : Module("ModalMapping", ctx, Source, "FieldsData", "ModelCreation"),
    mappingmethod_(ctx->subVar("mappingmethod")),
    integrationmethod_(ctx->subVar("integrationmethod")),
    integrationfilter_(ctx->subVar("integrationfilter")),
    def_value_(ctx->subVar("def-value"))  
{
}


void ModalMapping::execute()
{
  FieldHandle fsrc, fdst, fout;
  
  if (!(get_input_handle("Source",fsrc,true))) return;
  if (!(get_input_handle("Destination",fdst,true))) return;
  
  if (inputs_changed_ || mappingmethod_.changed() || integrationmethod_.changed() ||
      integrationfilter_.changed() || def_value_.changed() || !oport_cached("Destination"))
  {
    std::string mappingmethod = mappingmethod_.get();
    std::string integrationmethod = integrationmethod_.get();
    std::string integrationfilter = integrationfilter_.get();
    double def_value = def_value_.get();
    
    SCIRunAlgo::FieldsAlgo algo(this);
    
    if (!(algo.ModalMapping(fsrc,fdst,fout,mappingmethod,integrationmethod,integrationfilter,def_value))) return;
    
    send_output_handle("Destination",fout,false);
  }
}

} // End namespace ModelCreation


