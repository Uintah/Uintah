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
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class GetNrrdsFromBundle : public Module {
public:
  GetNrrdsFromBundle(GuiContext*);
  virtual void execute();
  
private:
  GuiString             guinrrd1name_;
  GuiString             guinrrd2name_;
  GuiString             guinrrd3name_;
  GuiInt                guitransposenrrd1_;
  GuiInt                guitransposenrrd2_;
  GuiInt                guitransposenrrd3_;  
  GuiString             guinrrds_;
};


DECLARE_MAKER(GetNrrdsFromBundle)

GetNrrdsFromBundle::GetNrrdsFromBundle(GuiContext* ctx)
  : Module("GetNrrdsFromBundle", ctx, Filter, "Bundle", "SCIRun"),
    guinrrd1name_(get_ctx()->subVar("nrrd1-name"), "nrrd1"),
    guinrrd2name_(get_ctx()->subVar("nrrd2-name"), "nrrd2"),
    guinrrd3name_(get_ctx()->subVar("nrrd3-name"), "nrrd3"),
    guitransposenrrd1_(get_ctx()->subVar("transposenrrd1"), 0),
    guitransposenrrd2_(get_ctx()->subVar("transposenrrd2"), 0),
    guitransposenrrd3_(get_ctx()->subVar("transposenrrd3"), 0),
    guinrrds_(get_ctx()->subVar("nrrd-selection"), "")
{
}


void
GetNrrdsFromBundle::execute()
{
  BundleHandle handle;
  
  // Get data from input port:
  if (!(get_input_handle("bundle",handle,true))) return; 
  
  if (inputs_changed_ || guinrrd1name_.changed() || guinrrd2name_.changed() ||
      guinrrd3name_.changed() || guitransposenrrd1_.changed() ||
      guitransposenrrd2_.changed() || guitransposenrrd3_.changed() ||
      !oport_cached("bundle") || !oport_cached("nrrd1") ||
      !oport_cached("nrrd2") || !oport_cached("nrrd3"))
  {
    NrrdDataHandle fhandle;
    
    std::string nrrd1name = guinrrd1name_.get();
    std::string nrrd2name = guinrrd2name_.get();
    std::string nrrd3name = guinrrd3name_.get();
    int transposenrrd1 = guitransposenrrd1_.get();
    int transposenrrd2 = guitransposenrrd2_.get();
    int transposenrrd3 = guitransposenrrd3_.get();
    std::string nrrdlist;
        
    int numNrrds = handle->numNrrds();
    for (int p = 0; p < numNrrds; p++)
    {
      nrrdlist += "{" + handle->getNrrdName(p) + "} ";
    }

    guinrrds_.set(nrrdlist);
    get_ctx()->reset();
  
    // We need to set bundle properties hence we need to detach
    handle.detach();
    
    // Send nrrd1 if we found one that matches the name:
    if (handle->isNrrd(nrrd1name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd1) handle->transposeNrrd(true);    
      fhandle = handle->getNrrd(nrrd1name);
      send_output_handle("nrrd1",fhandle,false);
    } 

    // Send nrrd2 if we found one that matches the name:
    if (handle->isNrrd(nrrd2name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd2) handle->transposeNrrd(true);    
      fhandle = handle->getNrrd(nrrd2name);
      send_output_handle("nrrd2",fhandle,false);
    } 

    // Send matrix3 if we found one that matches the name:
    if (handle->isNrrd(nrrd3name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd3) handle->transposeNrrd(true);    
      fhandle = handle->getNrrd(nrrd3name);
      send_output_handle("nrrd3",fhandle,false);
    } 
    send_output_handle("bundle",handle,false);    
  }
}

