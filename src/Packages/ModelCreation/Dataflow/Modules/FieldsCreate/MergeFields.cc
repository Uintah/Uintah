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
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Dataflow/Network/Module.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class MergeFields : public Module {
public:
  MergeFields(GuiContext*);

  virtual ~MergeFields();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiDouble guitolerance_;
  GuiInt    guimergenodes_;
  GuiInt    guiforcepointcloud_;

};


DECLARE_MAKER(MergeFields)
MergeFields::MergeFields(GuiContext* ctx)
  : Module("MergeFields", ctx, Source, "FieldsCreate", "ModelCreation"),
  guitolerance_(get_ctx()->subVar("tolerance")),
  guimergenodes_(get_ctx()->subVar("force-nodemerge")),
  guiforcepointcloud_(get_ctx()->subVar("force-pointcloud"))
{
}

MergeFields::~MergeFields(){
}

void MergeFields::execute()
{
  int numiports = num_input_ports()-1;
  
  std::vector<SCIRun::FieldHandle> fields(numiports);
  
  for (size_t p = 0; p < numiports; p++)
  {
    FieldIPort *field_iport;
    if (!(field_iport = dynamic_cast<FieldIPort *>(get_input_port(p))))
    {
      error("Could not find Field input port");
      return;
    }
    FieldHandle input;
    field_iport->get(input);
    
    if (input.get_rep() == 0)
    {
      warning("No Field was found on input port");
      return;
    }
    
    fields[p] = input;
  }
  
  FieldHandle output;

  double tolerance = 0.0;
  bool   mergenodes = false;
  bool   forcepointcloud = false;

  tolerance = guitolerance_.get();
  if (guimergenodes_.get()) mergenodes = true;
  if (guiforcepointcloud_.get()) forcepointcloud = true;

  FieldsAlgo fieldalgo(dynamic_cast<ProgressReporter *>(this));  

  if (!(fieldalgo.MergeFields(fields,output,tolerance,mergenodes)))
  {
    error("The MergeFields algorithm failed.");
    return;
  }

  if (forcepointcloud)
  {
    FieldHandle temp;
    if (!(fieldalgo.ToPointCloud(output,temp)))
    {
      error("The ToPointCloud algorithm failed");
      return;
    }
    output = temp;
  }
  
  FieldOPort* output_oport = dynamic_cast<FieldOPort *>(get_output_port(0));
  if (output_oport) output_oport->send(output);
}

void
 MergeFields::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


