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


/*
 *  ConvertIndicesToSelectionMask.cc:
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

class ConvertIndicesToSelectionMask : public Module {
public:
  ConvertIndicesToSelectionMask(GuiContext*);

  virtual ~ConvertIndicesToSelectionMask();

  virtual void execute();
};


DECLARE_MAKER(ConvertIndicesToSelectionMask)
ConvertIndicesToSelectionMask::ConvertIndicesToSelectionMask(GuiContext* ctx)
  : Module("ConvertIndicesToSelectionMask", ctx, Source, "SelectionMask", "ModelCreation")
{
}


ConvertIndicesToSelectionMask::~ConvertIndicesToSelectionMask()
{
}


void
ConvertIndicesToSelectionMask::execute()
{
  SCIRun::MatrixHandle idxmat;
  if (!get_input_handle("Indices", idxmat)) return;

  SCIRun::MatrixHandle lenmat;
  int size = -1;
  if (get_input_handle("Size", lenmat, false))
  {
    if (lenmat->get_data_size() > 0)
    {
      size = static_cast<int>(lenmat->get(0,0));
    }
  }

  SelectionMask mask(idxmat, size);
  
  if (!mask.isvalid())
  {
    error("Selectionmask could not be created");
    return;
  }
  
  MatrixHandle output = mask.gethandle();
  send_output_handle("SelectionMask", output);
}


} // End namespace CardioWave


