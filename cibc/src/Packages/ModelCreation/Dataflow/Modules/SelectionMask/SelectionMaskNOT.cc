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


/*
 *  SelectionMaskNOT.cc:
 *
 *  Written by:
 *   Jeroen Stinstra
 *
 */

#include <Packages/ModelCreation/Core/Fields/SelectionMask.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

namespace ModelCreation {

using namespace SCIRun;

class SelectionMaskNOT : public Module {
public:
  SelectionMaskNOT(GuiContext*);

  virtual ~SelectionMaskNOT();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(SelectionMaskNOT)
SelectionMaskNOT::SelectionMaskNOT(GuiContext* ctx)
  : Module("SelectionMaskNOT", ctx, Source, "SelectionMask", "ModelCreation")
{
}

SelectionMaskNOT::~SelectionMaskNOT()
{
}

void SelectionMaskNOT::execute()
{
  MatrixIPort *iport;
  MatrixOPort *oport;
  MatrixHandle input, output;
  
  if (!(iport = dynamic_cast<MatrixIPort *>(get_input_port(0))))
  {
    // nothing to do no ports available
    return;
  }
  
  if (!(iport->get(input)))
  {
    warning("No data could be found on the input ports");
    return;
  }

  SelectionMask mask(input);
  if (!mask.isvalid())
  {
    error("Data found on the first data port is not a valid selection mask");
    return;
  }
  
  SelectionMask newmask;
  newmask = mask.NOT();
  
  if(!newmask.isvalid())
  {
    error("Cannot NOT selection mask");
  }
  
  output = newmask.gethandle();
  
  if (!(oport = dynamic_cast<MatrixOPort *>(get_output_port(0))))
  {
    error("No output port is defined");
    return;
  }
  
  oport->send(output);
}

void SelectionMaskNOT::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace 

