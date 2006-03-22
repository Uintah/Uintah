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
 *  FieldDataElemToNode.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>

#include <Packages/ModelCreation/Core/Fields/FieldsAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class FieldDataElemToNode : public Module {
  public:
    FieldDataElemToNode(GuiContext*);

    virtual ~FieldDataElemToNode();

    virtual void execute();

    virtual void tcl_command(GuiArgs&, void*);
  private:
    GuiString method_;
    int fGeneration_;  
    std::string oldmethod_;
  
};


DECLARE_MAKER(FieldDataElemToNode)
FieldDataElemToNode::FieldDataElemToNode(GuiContext* ctx)
  : Module("FieldDataElemToNode", ctx, Source, "FieldsData", "ModelCreation"),
    method_(get_ctx()->subVar("method"))  
{
}

FieldDataElemToNode::~FieldDataElemToNode(){
}

void
 FieldDataElemToNode::execute()
{

  FieldIPort *field_iport;
  
  if (!(field_iport = dynamic_cast<FieldIPort *>(get_input_port(0))))
  {
    error("Could not find Field input port");
    return;
  }
   
  FieldHandle input;
  
  field_iport->get(input);
  
  if (input.get_rep()==0)
  {
    warning("No Field was found on input port");
    return;
  }
 
  bool update = false;

  if ((fGeneration_ != input->generation )||(oldmethod_ != method_.get())) {
    fGeneration_ = input->generation;
    oldmethod_ = method_.get();
    update = true;
  }
  
  std::string method = method_.get();

  if(update)
  {
    FieldsAlgo fieldmath(this);
    FieldHandle output;

    if(!(fieldmath.FieldDataElemToNode(input,output,method)))
    {
      return;
    }
  
    FieldOPort* output_oport = dynamic_cast<FieldOPort *>(get_output_port(0));
    if (output_oport) output_oport->send(output);
  }
}

void
 FieldDataElemToNode::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


