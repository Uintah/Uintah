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

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Packages/ModelCreation/Core/Fields/FieldsAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class ClipFieldBySelectionMask : public Module {
  public:
    ClipFieldBySelectionMask(GuiContext*);

    virtual ~ClipFieldBySelectionMask();

    virtual void execute();

    virtual void tcl_command(GuiArgs&, void*);

  private:
    int fGeneration_;
};


DECLARE_MAKER(ClipFieldBySelectionMask)
ClipFieldBySelectionMask::ClipFieldBySelectionMask(GuiContext* ctx)
  : Module("ClipFieldBySelectionMask", ctx, Source, "FieldsCreate", "ModelCreation"),
  fGeneration_(-1)
{
}

ClipFieldBySelectionMask::~ClipFieldBySelectionMask(){
}

void ClipFieldBySelectionMask::execute()
{
  FieldIPort *field_iport;
  MatrixIPort *selmask_iport;
  
  if (!(field_iport = dynamic_cast<FieldIPort *>(get_input_port(0))))
  {
    error("Could not find Field input port");
    return;
  }
  
  if (!(selmask_iport = dynamic_cast<MatrixIPort *>(get_input_port(1))))
  {
    error("Could not find SelectionMask input port");
    return;
  }
  
  FieldHandle input;
  MatrixHandle selmask;
  
  field_iport->get(input);
  selmask_iport->get(selmask);
  
  if (input.get_rep() == 0)
  {
    warning("No Field was found on input port");
    return;
  }
  
  if (selmask.get_rep() == 0)
  {
    warning("No SelectionMask was found on the input");
    return;
  }

  bool update = false;

  if( fGeneration_ != input->generation ) {
    fGeneration_ = input->generation;
    update = true;
  }

  if(update)
  {
    FieldsAlgo fieldmath(this);
  
    FieldHandle output;
    MatrixHandle interpolant;
    if(!(fieldmath.ClipFieldBySelectionMask(input,output,selmask,interpolant)))
    {
      return;
    }
  
    FieldOPort* field_oport = dynamic_cast<FieldOPort *>(get_output_port(0));
    if (field_oport) field_oport->send(output);

    MatrixOPort* interpolant_oport = dynamic_cast<MatrixOPort *>(get_output_port(1));
    if (interpolant_oport) interpolant_oport->send(interpolant);
  }
}

void
 ClipFieldBySelectionMask::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


