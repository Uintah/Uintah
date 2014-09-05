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

#include <Packages/ModelCreation/Core/Fields/FieldsAlgo.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>

namespace ModelCreation {

using namespace SCIRun;

class CompartmentBoundary : public Module {
public:
  CompartmentBoundary(GuiContext*);
  virtual void execute();
  
private:
  GuiInt    guiuserange_;
  GuiDouble guiminrange_;
  GuiDouble guimaxrange_;
  GuiInt    guiincludeouterboundary_;
  GuiInt    guiinnerboundaryonly_;
  
  
};


DECLARE_MAKER(CompartmentBoundary)
CompartmentBoundary::CompartmentBoundary(GuiContext* ctx)
  : Module("CompartmentBoundary", ctx, Source, "FieldsCreate", "ModelCreation"),
    guiuserange_(ctx->subVar("userange")),
    guiminrange_(ctx->subVar("minrange")),
    guimaxrange_(ctx->subVar("maxrange")),
    guiincludeouterboundary_(ctx->subVar("includeouterboundary")),
    guiinnerboundaryonly_(ctx->subVar("innerboundaryonly"))    
{
}

void CompartmentBoundary::execute()
{
  FieldIPort* iport = dynamic_cast<FieldIPort*>(get_iport(0));
  if (iport == 0) 
  {
    error("Could not find input port");
    return;
  }

  FieldOPort* oport = dynamic_cast<FieldOPort*>(get_oport(0));
  if (oport == 0) 
  {
    error("Could not find output port");
    return;
  }

  FieldHandle ifield, ofield;
  FieldsAlgo algo(dynamic_cast<ProgressReporter *>(this));

  double minrange, maxrange;
  bool   userange, includeouterboundary;
  bool   innerboundaryonly;

  minrange = guiminrange_.get();
  maxrange = guimaxrange_.get();
  userange = static_cast<bool>(guiuserange_.get());
  includeouterboundary = static_cast<bool>(guiincludeouterboundary_.get());
  innerboundaryonly = static_cast<bool>(guiinnerboundaryonly_.get());

  iport->get(ifield);
  if(algo.CompartmentBoundary(ifield,ofield,minrange,maxrange,userange,includeouterboundary,innerboundaryonly)) oport->send(ofield);
}

} // End namespace ModelCreation


