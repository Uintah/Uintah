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

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class LinkToCompGrid : public Module {
public:
  LinkToCompGrid(GuiContext*);
  virtual void execute();
};


DECLARE_MAKER(LinkToCompGrid)
LinkToCompGrid::LinkToCompGrid(GuiContext* ctx)
  : Module("LinkToCompGrid", ctx, Source, "FiniteElements", "ModelCreation")
{
}


void LinkToCompGrid::execute()
{
  MatrixHandle NodeLink;
  MatrixHandle GeomToComp, CompToGeom;
  
  if (!(get_input_handle("NodeLink",NodeLink,true))) return;
  
  FieldsAlgo algo(this);
  algo.LinkToCompGrid(NodeLink,GeomToComp,CompToGeom);

  send_output_handle("GeomToComp",GeomToComp,true);
  send_output_handle("CompToGeom",CompToGeom,true);
}


} // End namespace ModelCreation


