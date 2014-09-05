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
 *  ConvertSelectionMaskToIndices.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Packages/ModelCreation/Core/Fields/SelectionMask.h>

namespace ModelCreation {

using namespace SCIRun;

class ConvertSelectionMaskToIndices : public Module {
public:
  ConvertSelectionMaskToIndices(GuiContext*);

  virtual ~ConvertSelectionMaskToIndices();

  virtual void execute();
};


DECLARE_MAKER(ConvertSelectionMaskToIndices)
ConvertSelectionMaskToIndices::ConvertSelectionMaskToIndices(GuiContext* ctx)
  : Module("ConvertSelectionMaskToIndices", ctx, Source, "SelectionMask", "ModelCreation")
{
}


ConvertSelectionMaskToIndices::~ConvertSelectionMaskToIndices()
{
}


void
ConvertSelectionMaskToIndices::execute()
{
  SCIRun::MatrixHandle input;
  if (!get_input_handle("SelectionMask", input)) return;

  SelectionMask mask(input);
  
  if (!mask.isvalid())
  {
    error("Selection at input port is not valid");
    return;
  }
  
  SCIRun::MatrixHandle lenmat = scinew SCIRun::DenseMatrix(1, 1);
  lenmat->put(0, 0, static_cast<double>(mask.size()));
  
  SCIRun::MatrixHandle idxmat;
  mask.get_indices(idxmat);
  
  send_output_handle("Indices", idxmat);
  send_output_handle("Size", lenmat);
}


} // End namespace CardioWave


