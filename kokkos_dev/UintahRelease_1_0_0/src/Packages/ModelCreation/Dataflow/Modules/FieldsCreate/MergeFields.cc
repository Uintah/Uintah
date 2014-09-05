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

class MergeFields : public Module {
public:
  MergeFields(GuiContext*);
  virtual void execute();

private:
  GuiDouble guitolerance_;
  GuiInt    guimergenodes_;
  GuiInt    guiforcepointcloud_;
  GuiInt    guimatchval_;

};


DECLARE_MAKER(MergeFields)
MergeFields::MergeFields(GuiContext* ctx)
  : Module("MergeFields", ctx, Source, "FieldsCreate", "ModelCreation"),
  guitolerance_(get_ctx()->subVar("tolerance")),
  guimergenodes_(get_ctx()->subVar("force-nodemerge")),
  guiforcepointcloud_(get_ctx()->subVar("force-pointcloud")),
  guimatchval_(get_ctx()->subVar("matchval"))
{
}

void MergeFields::execute()
{
  std::vector<SCIRun::FieldHandle> fields;
  if(!(get_dynamic_input_handles("Field",fields,true))) return;

  FieldHandle output;

  double tolerance = 0.0;
  bool   mergenodes = false;
  bool   forcepointcloud = false;
  bool   matchval = false;
  
  tolerance = guitolerance_.get();
  if (guimergenodes_.get()) mergenodes = true;
  if (guiforcepointcloud_.get()) forcepointcloud = true;
  if (guimatchval_.get()) matchval = true;

  SCIRunAlgo::FieldsAlgo algo(this);  

  if (!(algo.MergeFields(fields,output,tolerance,mergenodes,true,matchval))) return;

  if (forcepointcloud)
  {
    FieldHandle temp;
    if (!(algo.ToPointCloud(output,temp))) return;
    output = temp;
  }
  
  send_output_handle("Field",output,true);
}


} // End namespace ModelCreation


