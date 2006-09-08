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

#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>

namespace ModelCreation {

using namespace SCIRun;

class LinkFieldBoundary : public Module {
public:
  LinkFieldBoundary(GuiContext*);
  virtual void execute();
  
private:
  GuiInt guilinkx_;
  GuiInt guilinky_;
  GuiInt guilinkz_;
  GuiDouble guitol_;
  
};


DECLARE_MAKER(LinkFieldBoundary)
LinkFieldBoundary::LinkFieldBoundary(GuiContext* ctx)
  : Module("LinkFieldBoundary", ctx, Source, "FieldsGeometry", "ModelCreation"),
  guilinkx_(get_ctx()->subVar("linkx")),
  guilinky_(get_ctx()->subVar("linky")),
  guilinkz_(get_ctx()->subVar("linkz")),
  guitol_(get_ctx()->subVar("tol"))
{
}


void
LinkFieldBoundary::execute()
{
  FieldHandle input, output;
  MatrixHandle NodeLink, ElemLink;
  
  if(!(get_input_handle("Field",input,true))) return;

  if (inputs_changed_ ||  guilinkx_.changed() || guilinky_.changed() ||
      guilinkz_.changed() || guitol_.changed() || !oport_cached("Field") ||
      !oport_cached("NodeLink") || !oport_cached("ElemLink"))
  {
    const double tol = guitol_.get();
    const bool   linkx = guilinkx_.get();
    const bool   linky = guilinky_.get();
    const bool   linkz = guilinkz_.get();

    SCIRunAlgo::FieldsAlgo fieldsalgo(this);
    if(!(fieldsalgo.LinkFieldBoundary(input,NodeLink,ElemLink,tol,linkx,linky,linkz))) return;
    
    output = input->clone();
    output->set_property("NodeLink",NodeLink,false);
    output->set_property("ElemLink",ElemLink,false);
    
    send_output_handle("Field",output,false);
    send_output_handle("NodeLink",NodeLink,false);
    send_output_handle("ElemLink",ElemLink,false);  
  }
}

} // End namespace ModelCreation


