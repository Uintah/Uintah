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
 *  SelectionMaskToIndices.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Packages/ModelCreation/Core/Fields/SelectionMask.h>

namespace ModelCreation {

using namespace SCIRun;

class SelectionMaskToIndices : public Module {
public:
  SelectionMaskToIndices(GuiContext*);

  virtual ~SelectionMaskToIndices();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(SelectionMaskToIndices)
SelectionMaskToIndices::SelectionMaskToIndices(GuiContext* ctx)
  : Module("SelectionMaskToIndices", ctx, Source, "SelectionMask", "ModelCreation")
{
}

SelectionMaskToIndices::~SelectionMaskToIndices(){
}

void SelectionMaskToIndices::execute()
{
  SCIRun::MatrixIPort *selection_iport;
  SCIRun::MatrixOPort *indices_oport;
  SCIRun::MatrixOPort *length_oport;
  
  SCIRun::MatrixHandle input;
  
  if (!(selection_iport = dynamic_cast<SCIRun::MatrixIPort *>(getIPort(0))))
  {
    error("Could not find input port for selection mask");
    return;
  }

  if (!(indices_oport = dynamic_cast<SCIRun::MatrixOPort *>(getOPort(0))))
  {
    error("Could not find output port for indices matrix");
    return;
  }

  if (!(length_oport = dynamic_cast<SCIRun::MatrixOPort *>(getOPort(1))))
  {
    error("Could not find output port for length matrix");
    return;
  }

  if (!(selection_iport->get(input)))
  {
    warning("No data could be found on the input ports");
    return;
  }

  SelectionMask mask(input);
  
  if (!mask.isvalid())
  {
    error("Selection at input port is not valid");
    return;
  }
  
  SCIRun::MatrixHandle lenmat = dynamic_cast<SCIRun::Matrix *>(scinew SCIRun::DenseMatrix(1,1));
  if (lenmat.get_rep() == 0)
  {
    error("Could not allocate enough memory");
    return;
  }
  lenmat->put(0,0,static_cast<double>(mask.size()));
  
  SCIRun::MatrixHandle idxmat;
  mask.get_indices(idxmat);
  
  indices_oport->send(idxmat);
  length_oport->send(lenmat);
}

void
 SelectionMaskToIndices::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


