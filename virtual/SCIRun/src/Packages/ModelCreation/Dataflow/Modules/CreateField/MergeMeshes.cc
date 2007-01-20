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

#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Dataflow/Network/Module.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class MergeMeshes : public Module {
public:
  MergeMeshes(GuiContext*);
  virtual void execute();

private:
  GuiDouble guitolerance_;
  GuiInt    guimergenodes_;
  GuiInt    guiforcepointcloud_;
  GuiInt    guimatchval_;

};


DECLARE_MAKER(MergeMeshes)
MergeMeshes::MergeMeshes(GuiContext* ctx)
  : Module("MergeMeshes", ctx, Source, "CreateField", "ModelCreation"),
  guitolerance_(get_ctx()->subVar("tolerance")),
  guimergenodes_(get_ctx()->subVar("force-nodemerge")),
  guiforcepointcloud_(get_ctx()->subVar("force-pointcloud")),
  guimatchval_(get_ctx()->subVar("matchval"))
{
}

void MergeMeshes::execute()
{
  // Define local handles of data objects:
  std::vector<SCIRun::FieldHandle> fields;
  FieldHandle output;

  // Get the new input data:  
  if(!(get_dynamic_input_handles("Field",fields,true))) return;

  // Only reexecute if the input changed. SCIRun uses simple scheduling
  // that executes every module downstream even if no data has changed: 
  if (inputs_changed_ ||  guitolerance_.changed() || guimergenodes_.changed()  || !oport_cached("Field"))
  {
    double tolerance = 0.0;
    bool   mergenodes = false;
  
    tolerance = guitolerance_.get();
    if (guimergenodes_.get()) mergenodes = true;

    SCIRunAlgo::FieldsAlgo algo(this);  
    if (!(algo.MergeMeshes(fields,output,tolerance,mergenodes,true))) return;

    // send new output if there is any:        
    send_output_handle("Field",output,false);
  }
}


} // End namespace ModelCreation


