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
 *  SelectionMaskXOR.cc:
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

class SelectionMaskXOR : public Module {
public:
  SelectionMaskXOR(GuiContext*);

  virtual ~SelectionMaskXOR();

  virtual void execute();
};


DECLARE_MAKER(SelectionMaskXOR)
SelectionMaskXOR::SelectionMaskXOR(GuiContext* ctx)
  : Module("SelectionMaskXOR", ctx, Source, "SelectionMask", "ModelCreation")
{
}


SelectionMaskXOR::~SelectionMaskXOR()
{
}


void
SelectionMaskXOR::execute()
{
  MatrixHandle input;
  if (!get_input_handle("SelectionMask1", input)) return;

  SelectionMask mask(input);
  if (!mask.isvalid())
  {
    error("Data found on the first data port is not a valid selection mask");
    return;
  }
  
  if (!get_input_handle("SelectionMask2", input)) return;

  SelectionMask mask2(input);
  if (!mask2.isvalid())
  {
    error("Data found on the second data port is not a valid selection mask");
    return;
  }
  
  SelectionMask newmask;
  
  newmask = mask.XOR(mask2); 
  
  if(!newmask.isvalid())
  {
    error("Cannot perform an XOR operation on the two selection vectors, mask sure they are of the same length");
  }
  
  MatrixHandle output = newmask.gethandle();
  send_output_handle("SelectionMask", output);
}


} // End namespace CardioWave


