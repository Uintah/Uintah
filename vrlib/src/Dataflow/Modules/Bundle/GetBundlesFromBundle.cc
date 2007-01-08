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

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;

class GetBundlesFromBundle : public Module {
public:
  GetBundlesFromBundle(GuiContext*);
  virtual void execute();
  
private:
  GuiString             guibundle1name_;
  GuiString             guibundle2name_;
  GuiString             guibundle3name_;
  GuiString             guibundles_;
};


DECLARE_MAKER(GetBundlesFromBundle)

GetBundlesFromBundle::GetBundlesFromBundle(GuiContext* ctx)
  : Module("GetBundlesFromBundle", ctx, Filter, "Bundle", "SCIRun"),
    guibundle1name_(get_ctx()->subVar("bundle1-name"), "bundle1"),
    guibundle2name_(get_ctx()->subVar("bundle2-name"), "bundle2"),
    guibundle3name_(get_ctx()->subVar("bundle3-name"), "bundle3"),
    guibundles_(get_ctx()->subVar("bundle-selection"), "")
{
}

void GetBundlesFromBundle::execute()
{
  // Define input handle:
  BundleHandle handle;
  
  // Get data from input port:
  if (!(get_input_handle("bundle",handle,true))) return;
  
  if (inputs_changed_ || guibundle1name_.changed() || guibundle2name_.changed() ||
      guibundle3name_.changed() || !oport_cached("bundle") || !oport_cached("bundle1") ||
       !oport_cached("bundle2") || !oport_cached("bundle3"))
  {
    BundleHandle fhandle;
    std::string bundle1name = guibundle1name_.get();
    std::string bundle2name = guibundle2name_.get();
    std::string bundle3name = guibundle3name_.get();
    std::string bundlelist;
    
    int numBundles = handle->numBundles();
    for (int p = 0; p < numBundles; p++)
    {
      bundlelist += "{" + handle->getBundleName(p) + "} ";
    }

    guibundles_.set(bundlelist);
    get_ctx()->reset();  
  
    // Send bundle1 if we found one that matches the name:
    if (handle->isBundle(bundle1name))
    {
      fhandle = handle->getBundle(bundle1name);
      send_output_handle("bundle1",fhandle,false);
    } 

    // Send bundle2 if we found one that matches the name:
    if (handle->isBundle(bundle2name))
    {
      fhandle = handle->getBundle(bundle2name);
      send_output_handle("bundle2",fhandle,false);
    } 

    // Send bundle3 if we found one that matches the name:  
    if (handle->isBundle(bundle3name))
    {
      fhandle = handle->getBundle(bundle3name);
      send_output_handle("bundle3",fhandle,false);
    } 
    
    send_output_handle("bundle",handle,false);
  }
}


