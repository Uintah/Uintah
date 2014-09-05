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
 *  BundleGetNrrd.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdString.h>

using namespace SCIRun;
using namespace std;

class BundleGetNrrd : public Module {
public:
  BundleGetNrrd(GuiContext*);

  virtual ~BundleGetNrrd();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guinrrd1name_;
  GuiString             guinrrd2name_;
  GuiString             guinrrd3name_;
  GuiInt                guitransposenrrd1_;
  GuiInt                guitransposenrrd2_;
  GuiInt                guitransposenrrd3_;  
  GuiString             guinrrds_;
};


DECLARE_MAKER(BundleGetNrrd)
  BundleGetNrrd::BundleGetNrrd(GuiContext* ctx)
    : Module("BundleGetNrrd", ctx, Source, "Bundle", "SCIRun"),
      guinrrd1name_(ctx->subVar("nrrd1-name")),
      guinrrd2name_(ctx->subVar("nrrd2-name")),
      guinrrd3name_(ctx->subVar("nrrd3-name")),
      guitransposenrrd1_(ctx->subVar("transposenrrd1")),
      guitransposenrrd2_(ctx->subVar("transposenrrd2")),
      guitransposenrrd3_(ctx->subVar("transposenrrd3")),
      guinrrds_(ctx->subVar("nrrd-selection"))
{

}

BundleGetNrrd::~BundleGetNrrd(){
}


void
BundleGetNrrd::execute()
{
  string nrrd1name = guinrrd1name_.get();
  string nrrd2name = guinrrd2name_.get();
  string nrrd3name = guinrrd3name_.get();
  int transposenrrd1 = guitransposenrrd1_.get();
  int transposenrrd2 = guitransposenrrd2_.get();
  int transposenrrd3 = guitransposenrrd3_.get();
  string nrrdlist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  NrrdDataHandle fhandle;
  NrrdOPort *ofport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("bundle"))))
    {
      error("Could not find bundle input port");
      return;
    }

  if (!(iport->get(handle)))
    {   
      warning("No bundle connected to the input port");
      return;
    }

  if (handle.get_rep() == 0)
    {   
      warning("Empty bundle connected to the input port");
      return;
    }


  int numNrrds = handle->numNrrds();
  for (int p = 0; p < numNrrds; p++)
    {
      nrrdlist += "{" + handle->getNrrdName(p) + "} ";
    }

  guinrrds_.set(nrrdlist);
  ctx->reset();

 
  if (!(ofport = static_cast<NrrdOPort *>(get_oport("nrrd1"))))
    {
      error("Could not find nrrd 1 output port");
      return; 
    }
 
  NrrdIPort *niport = static_cast<NrrdIPort *>(getIPort("name1"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          nrrd1name = nrrdstring.getstring();
          guinrrd1name_.set(nrrd1name);
          ctx->reset();
        }
    } 
 
  if (handle->isNrrd(nrrd1name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd1) handle->transposeNrrd(true);    
      fhandle = handle->getNrrd(nrrd1name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<NrrdOPort *>(get_oport("nrrd2"))))
    {
      error("Could not find nrrd 2 output port");
      return; 
    }
  
  niport = static_cast<NrrdIPort *>(getIPort("name2"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          nrrd2name = nrrdstring.getstring();
          guinrrd2name_.set(nrrd2name);
          ctx->reset();
        }
    } 
    
  if (handle->isNrrd(nrrd2name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd2) handle->transposeNrrd(true);
      fhandle = handle->getNrrd(nrrd2name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<NrrdOPort *>(get_oport("nrrd3"))))
    {
      error("Could not find nrrd 3 output port");
      return; 
    }
 
 
  niport = static_cast<NrrdIPort *>(getIPort("name3"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          nrrd3name = nrrdstring.getstring();
          guinrrd3name_.set(nrrd3name);
          ctx->reset();
        }
    } 
 
  if (handle->isNrrd(nrrd3name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd3) handle->transposeNrrd(true);    
      fhandle = handle->getNrrd(nrrd3name);
      ofport->send(fhandle);
    }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
    {
      oport->send(handle);
    }
        
}

void
BundleGetNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




