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
 *  MatrixReader.cc: Read a persistent matrix from a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Datatypes/NrrdString.h>
#include <sys/stat.h>

namespace CardioWave {

using namespace SCIRun;

class FileSelector: public Module {
private:
  GuiString       gui_types_;
  GuiString       gui_filetype_;
  GuiString       filename_;
  NrrdDataHandle  handle_;

public:
  FileSelector(GuiContext* ctx);
  virtual ~FileSelector();
  virtual void execute();
  
};

DECLARE_MAKER(FileSelector)

FileSelector::FileSelector(GuiContext* ctx)
  : Module("FileSelector", ctx, Source, "Tools", "CardioWave"),
    gui_types_(ctx->subVar("types")),
    gui_filetype_(ctx->subVar("filetype")),
    filename_(ctx->subVar("filename"))
{
  string importtypes = "{";
  importtypes += "{{All files} {.*} } ";
  importtypes += "}";
  gui_types_.set(importtypes);
  
  
}

FileSelector::~FileSelector()
{
}

void FileSelector::execute()
{

 const string fn(filename_.get());

  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if( fn == "" ) 
  {
    error("No file has been selected.  Please choose a file.");
    return;
  } 
  else if (stat(fn.c_str(), &buf)) 
  {
    error("File '" + fn + "' not found.");
    return;
  }

  NrrdString ns(fn);
  handle_ = ns.gethandle();
  
  NrrdOPort *oport = (NrrdOPort *)getOPort(0);
  if (!oport) 
  {
    error("Unable to initialize output port.");
    return;
  }
  
  oport->send(handle_);
}


} // End namespace SCIRun
