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
 *  SplitFileName.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Ports/StringPort.h>

namespace SCIRun {

using namespace SCIRun;

class SplitFileName : public Module {
public:
  SplitFileName(GuiContext*);

  virtual ~SplitFileName();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(SplitFileName)
SplitFileName::SplitFileName(GuiContext* ctx)
  : Module("SplitFileName", ctx, Source, "String", "SCIRun")
{
}

SplitFileName::~SplitFileName(){
}

void
 SplitFileName::execute()
{
  StringIPort *iport;
  StringOPort *pn_oport, *fn_oport, *ext_oport;
  
  if(!(iport = dynamic_cast<StringIPort*>(get_iport(0))))
  {
    error("Could not find input port");
    return;
  }
  if(!(pn_oport = dynamic_cast<StringOPort*>(get_oport(0))))
  {
    error("Could not find pathname output port");
    return;
  }
  if(!(fn_oport = dynamic_cast<StringOPort*>(get_oport(1))))
  {
    error("Could not find filename output port");
    return;
  }
  if(!(ext_oport = dynamic_cast<StringOPort*>(get_oport(2))))
  {
    error("Could not find extension output port");
    return;
  }
  
  StringHandle filenameH;
  StringHandle fnH, pnH, extH;
  std::string filename, fn, pn, ext;
  iport->get(filenameH);
  
  if (filenameH.get_rep() == 0)
  {
    error("No input string was given at input port");
    return;
  }

  char sep = '/';
  char dot = '.';
  
  filename = filenameH->get();
  
  int lastsep = -1;
  for (size_t p = 0; p < filename.size(); p++) if (filename[p] == sep) lastsep = (int)p;
  
  if (lastsep > -1)
  {
    pn = filename.substr(0,lastsep+1);
    fn = filename.substr(lastsep+1);
  }
  else
  {
    pn = "";
    fn = filename;
  }

  int lastdot = -1;
  for (size_t p = 0; p < fn.size(); p++) if (fn[p] == dot) lastdot = (int)p;
  
  if (lastdot > -1)
  {
    ext = fn.substr(lastdot);
    fn = fn.substr(0,lastdot);
  }
  else
  {
    ext = "";
  }

  pnH = dynamic_cast<String *>(scinew String(pn));
  fnH = dynamic_cast<String *>(scinew String(fn));
  extH = dynamic_cast<String *>(scinew String(ext));

  pn_oport->send(pnH);
  fn_oport->send(fnH);
  ext_oport->send(extH);
}

void
 SplitFileName::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


