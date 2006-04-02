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

#include <Packages/ModelCreation/Core/Fields/FieldsAlgo.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class LinkFieldBoundaryByElement : public Module {
public:
  LinkFieldBoundaryByElement(GuiContext*);
  virtual void execute();
  
private:
  GuiInt guilinkx_;
  GuiInt guilinky_;
  GuiInt guilinkz_;
  GuiDouble guitol_;
  
};


DECLARE_MAKER(LinkFieldBoundaryByElement)
LinkFieldBoundaryByElement::LinkFieldBoundaryByElement(GuiContext* ctx)
  : Module("LinkFieldBoundaryByElement", ctx, Source, "FiniteElements", "ModelCreation"),
  guilinkx_(ctx->subVar("linkx")),
  guilinky_(ctx->subVar("linky")),
  guilinkz_(ctx->subVar("linkz")),
  guitol_(ctx->subVar("tol"))
{
}

void LinkFieldBoundaryByElement::execute()
{
  FieldHandle Field;
  MatrixHandle GeomToComp, CompToGeom, DomainLink, MembraneLink;
  if(!(get_input_handle("Field",Field,true))) return;

  double tol = guitol_.get();
  bool   linkx = static_cast<bool>(guilinkx_.get());
  bool   linky = static_cast<bool>(guilinky_.get());
  bool   linkz = static_cast<bool>(guilinkz_.get());

  FieldsAlgo fieldsalgo(this);
  if(!(fieldsalgo.LinkFieldBoundaryByElement(Field,GeomToComp,CompToGeom,DomainLink,MembraneLink,tol,linkx,linky,linkz))) return;
  
  send_output_handle("GeomToComp",GeomToComp,true);
  send_output_handle("CompToGeom",CompToGeom,true);  
  send_output_handle("DomainLink",DomainLink,true);  
  send_output_handle("MembraneLink",MembraneLink,true);  

}

} // End namespace ModelCreation


