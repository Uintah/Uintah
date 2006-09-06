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

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class BundleMerge : public Module {
public:
  BundleMerge(GuiContext*);
  virtual void execute();
};


DECLARE_MAKER(BundleMerge)
  BundleMerge::BundleMerge(GuiContext* ctx)
    : Module("BundleMerge", ctx, Source, "Bundle", "SCIRun")
{
}

void BundleMerge::execute()
{
  std::vector<BundleHandle> inputs;
  
  get_dynamic_input_handles("bundle",inputs,false);

  if (inputs_changed_ || !oport_cached("bundle"))
  {
    BundleHandle output;
    output = scinew Bundle;
  
    if (output.get_rep() == 0)
    {
      error("Could not allocate new bundle");
      return;
    }


    for (size_t p=0; p<inputs.size();p++)
    {
      output->merge(inputs[p]);
    }
    
    send_output_handle("bundle",output,false);
  }
}

