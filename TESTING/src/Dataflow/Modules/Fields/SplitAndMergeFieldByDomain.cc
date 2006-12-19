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
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>

namespace SCIRun {

class SplitAndMergeFieldByDomain : public Module {
public:
  SplitAndMergeFieldByDomain(GuiContext*);
  virtual void execute();

};


DECLARE_MAKER(SplitAndMergeFieldByDomain)
SplitAndMergeFieldByDomain::SplitAndMergeFieldByDomain(GuiContext* ctx)
  : Module("SplitAndMergeFieldByDomain", ctx, Source, "NewField", "SCIRun")
{
}


void SplitAndMergeFieldByDomain::execute()
{
  // Define local handles of data objects:
  FieldHandle input;
  FieldHandle output;
  
  // Get the new input data:    
  if(!(get_input_handle("Field",input,true))) return;

  // Only reexecute if the input changed. SCIRun uses simple scheduling
  // that executes every module downstream even if no data has changed:  
  if (inputs_changed_ || !oport_cached("SplitField"))
  {
    SCIRunAlgo::FieldsAlgo algo(this);
    if(!(algo.SplitAndMergeFieldByDomain(input,output))) return;

    // send new output if there is any:   
    send_output_handle("SplitField",output,false);
  }
}



} // End namespace SCIRun

