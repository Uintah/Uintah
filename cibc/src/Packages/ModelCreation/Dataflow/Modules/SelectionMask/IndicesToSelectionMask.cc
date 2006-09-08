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
 *  IndicesToSelectionMask.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Packages/ModelCreation/Core/Fields/SelectionMask.h>


namespace ModelCreation {

using namespace SCIRun;

class IndicesToSelectionMask : public Module {
public:
  IndicesToSelectionMask(GuiContext*);

  virtual ~IndicesToSelectionMask();

  virtual void execute();
};


DECLARE_MAKER(IndicesToSelectionMask)
IndicesToSelectionMask::IndicesToSelectionMask(GuiContext* ctx)
  : Module("IndicesToSelectionMask", ctx, Source, "SelectionMask", "ModelCreation")
{
}


IndicesToSelectionMask::~IndicesToSelectionMask()
{
}


void
IndicesToSelectionMask::execute()
{
  SCIRun::MatrixOPort *selection_oport;
  SCIRun::MatrixIPort *indices_iport;
  SCIRun::MatrixIPort *length_iport;
  
  SCIRun::MatrixHandle output;
  SCIRun::MatrixHandle lenmat;
  SCIRun::MatrixHandle idxmat;
  
  if (!(selection_oport = dynamic_cast<SCIRun::MatrixOPort *>(get_output_port(0))))
  {
    error("Could not find output port for selection mask");
    return;
  }

  if (!(indices_iport = dynamic_cast<SCIRun::MatrixIPort *>(get_input_port(0))))
  {
    error("Could not find input port for indices matrix");
    return;
  }

  if (!(length_iport = dynamic_cast<SCIRun::MatrixIPort *>(get_input_port(1))))
  {
    error("Could not find input port for length matrix");
    return;
  }

  if (!(indices_iport->get(idxmat)))
  {
    warning("No data could be found on the indices input port");
    return;
  }

  if (!(length_iport->get(lenmat)))
  {
    warning("No data could be found on the length input port, using maximum index as size of selection vector");
  }

  if (idxmat.get_rep() == 0)
  {
    error("Empty matrix on indices input");
    return;
  }
  
  int size = -1;

  if (lenmat.get_rep())
  {
    if (lenmat->get_data_size() > 0)
    {
      size = static_cast<int>(lenmat->get(0,0));
    }
  }

  SelectionMask mask(idxmat,size);
  
  if (!mask.isvalid())
  {
    error("Selectionmask could not be created");
    return;
  }
  
  output = mask.gethandle();
  selection_oport->send(output);
}


} // End namespace CardioWave


