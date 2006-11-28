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
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Dataflow/Network/Module.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class JoinFields : public Module {
public:
  JoinFields(GuiContext*);
  virtual void execute();

private:
  GuiInt    guiaccumulating_;
  GuiInt    guiclear_;
  GuiDouble guitolerance_;
  GuiInt    guimergenodes_;
  GuiInt    guiforcepointcloud_;
  GuiInt    guimatchval_;
  
  FieldHandle fHandle_;
};


DECLARE_MAKER(JoinFields)

JoinFields::JoinFields(GuiContext* ctx)
  : Module("JoinFields", ctx, Source, "NewField", "SCIRun"),
  guiaccumulating_(get_ctx()->subVar("accumulating"), 0),
  guiclear_(get_ctx()->subVar("clear", false), 0),  
  guitolerance_(get_ctx()->subVar("tolerance"), 0.0001),
  guimergenodes_(get_ctx()->subVar("force-nodemerge"),1),
  guiforcepointcloud_(get_ctx()->subVar("force-pointcloud"),0),
  guimatchval_(get_ctx()->subVar("matchval"),0)
{
}


void
JoinFields::execute()
{
  // Define local handles of data objects:
  std::vector<SCIRun::FieldHandle> fields;
  FieldHandle output;

  // Some stuff for old power apps
  if (guiclear_.get())
  {
    guiclear_.set(0);
    fHandle_ = 0;

    // Sending 0 does not clear caches.
    typedef PointCloudMesh<ConstantBasis<double> > PCMesh; 
    typedef NoDataBasis<double> NDBasis;
    typedef GenericField<PCMesh, NDBasis, vector<double> > PCField;
    FieldHandle handle = scinew PCField(scinew PCMesh());

    send_output_handle("Output Field", handle);
    return;
  }

  // Get the new input data:  
  if(!(get_dynamic_input_handles("Field",fields,true))) return;

  if (guiaccumulating_.get() && fHandle_.get_rep()) // appending fields
  {
    fields.push_back(fHandle_);
  }

  // Only reexecute if the input changed. SCIRun uses simple scheduling
  // that executes every module downstream even if no data has changed: 
  if (inputs_changed_ ||  guitolerance_.changed() ||
      guimergenodes_.changed() || guiforcepointcloud_.changed() ||
      guimatchval_.changed() || !oport_cached("Output Field"))
  {
    double tolerance = 0.0;
    bool   mergenodes = false;
    bool   forcepointcloud = false;
    bool   matchval = false;
  
    tolerance = guitolerance_.get();
    if (guimergenodes_.get()) mergenodes = true;
    if (guiforcepointcloud_.get()) forcepointcloud = true;
    if (guimatchval_.get()) matchval = true;

    SCIRunAlgo::FieldsAlgo algo(this);  
    if (!(algo.MergeFields(fields,output,tolerance,mergenodes,true,matchval)))
      return;

    // This option is here to be compatible with the old GatherFields module:
    // This is a separate algorithm now
    if (forcepointcloud)
    {
      if (!(algo.ConvertMeshToPointCloud(output,output))) return;
    }

    // send new output if there is any:        
    send_output_handle("Output Field",output,false);
  }
}

} // End namespace SCIRun

