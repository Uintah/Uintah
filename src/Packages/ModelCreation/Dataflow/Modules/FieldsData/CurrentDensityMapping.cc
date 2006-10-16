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
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>


namespace ModelCreation {

using namespace SCIRun;

class CurrentDensityMapping : public Module {
public:
  CurrentDensityMapping(GuiContext*);
  virtual void execute();

  private:
    GuiString mappingmethod_;
    GuiString integrationmethod_;
    GuiString integrationfilter_;
    GuiInt multiply_with_normal_;
    GuiInt calcnorm_;

};


DECLARE_MAKER(CurrentDensityMapping)
CurrentDensityMapping::CurrentDensityMapping(GuiContext* ctx)
  : Module("CurrentDensityMapping", ctx, Source, "FieldsData", "ModelCreation"),
    mappingmethod_(ctx->subVar("mappingmethod")),
    integrationmethod_(ctx->subVar("integrationmethod")),
    integrationfilter_(ctx->subVar("integrationfilter")),
    multiply_with_normal_(ctx->subVar("multiply-with-normal")),
    calcnorm_(ctx->subVar("calcnorm"))
{
}

void CurrentDensityMapping::execute()
{
  // Define input handles:
  FieldHandle fpot, fcon, fdst, fout;
  
  // Get input from ports:
  if (!(get_input_handle("Potential",fpot,true))) return;
  if (!(get_input_handle("Conductivity",fcon,true))) return;
  if (!(get_input_handle("Destination",fdst,true))) return;
  
  if (inputs_changed_ || mappingmethod_.changed() || integrationfilter_.changed() ||
      integrationmethod_.changed() || multiply_with_normal_.changed() ||
      !oport_cached("Destination"))
  {
    // Get parameters from GUI:
    std::string mappingmethod = mappingmethod_.get();
    std::string integrationmethod = integrationmethod_.get();
    std::string integrationfilter = integrationfilter_.get();
    bool multiply_with_normal = multiply_with_normal_.get();
    bool calcnorm = calcnorm_.get();
    
    // Entry point to fields algo library:
    SCIRunAlgo::FieldsAlgo algo(this);
    if (!(algo.CurrentDensityMapping(fpot,fcon,fdst,fout,mappingmethod,integrationmethod,integrationfilter,multiply_with_normal,calcnorm))) return;
    
    // Send output downstream:
    send_output_handle("Destination",fout,false);
  }
}


} // End namespace ModelCreation


