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
 *  SignedDistanceToField.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class SignedDistanceToField : public Module {
public:
  SignedDistanceToField(GuiContext*);

  virtual ~SignedDistanceToField();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(SignedDistanceToField)
SignedDistanceToField::SignedDistanceToField(GuiContext* ctx)
  : Module("SignedDistanceToField", ctx, Source, "FieldsData", "ModelCreation")
{
}

SignedDistanceToField::~SignedDistanceToField(){
}

void SignedDistanceToField::execute()
{
  error("This function is under development and not yet implemented");
}

void
 SignedDistanceToField::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


