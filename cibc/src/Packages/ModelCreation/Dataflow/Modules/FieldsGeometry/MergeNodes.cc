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

#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Dataflow/Network/Module.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class MergeNodes : public Module {
public:
  MergeNodes(GuiContext*);
  virtual void execute();
  
private:
  GuiDouble guitolerance_;
  GuiInt    guimatchval_;  
};


DECLARE_MAKER(MergeNodes)
MergeNodes::MergeNodes(GuiContext* ctx)
  : Module("MergeNodes", ctx, Source, "FieldsGeometry", "ModelCreation"),
  guitolerance_(get_ctx()->subVar("tolerance")),
  guimatchval_(get_ctx()->subVar("matchval"))
{
}

void MergeNodes::execute()
{
  FieldHandle input;
  if(!(get_input_handle("Field",input,true))) return;

  FieldHandle output;

  double tolerance = 0.0;
  bool   matchval = false;
  
  tolerance = guitolerance_.get();
  if (guimatchval_.get()) matchval = true;

  SCIRunAlgo::FieldsAlgo algo(this);  

  std::vector<FieldHandle> fields(1);
  fields[0] = input;
  if (!(algo.MergeFields(fields,output,tolerance,true,true,matchval))) return;

  send_output_handle("Field",output,false);
}

} // End namespace ModelCreation


