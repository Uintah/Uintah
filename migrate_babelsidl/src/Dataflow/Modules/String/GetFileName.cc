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
 *  GetFileName.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/StringPort.h>

namespace SCIRun {

using namespace SCIRun;

class GetFileName : public Module {
public:
  GetFileName(GuiContext*);

  virtual ~GetFileName();

  virtual void execute();

private:
  GuiString gui_filename_;
  GuiString gui_filebase_;
  GuiInt gui_delay_;

  StringHandle string_output_handle_;
};


DECLARE_MAKER(GetFileName)
GetFileName::GetFileName(GuiContext* ctx)
  : Module("GetFileName", ctx, Source, "String", "SCIRun"),
    gui_filename_(get_ctx()->subVar("filename"), ""),
    gui_filebase_(get_ctx()->subVar("filebase"), ""),
    gui_delay_(get_ctx()->subVar("delay"), 500),
    string_output_handle_(0)
{
}

GetFileName::~GetFileName(){
}

void
GetFileName::execute()
{
  if( gui_filename_.changed() ||
      !string_output_handle_.get_rep() ) {

    string_output_handle_ = StringHandle(scinew String(gui_filename_.get()));
  }

  send_output_handle( "Full filename", string_output_handle_, true );
}

} // End namespace SCIRun
