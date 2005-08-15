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
 *  BundleReader.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/DataIO/GenericReader.h>

using namespace SCIRun;
using namespace std;

class BundleReader : public GenericReader<BundleHandle> {
public:
  BundleReader(GuiContext*);

  virtual ~BundleReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
protected:
  GuiString guiTypes_;
  GuiString guiFileType_;
  
};


DECLARE_MAKER(BundleReader)
  BundleReader::BundleReader(GuiContext* ctx)
    : GenericReader<BundleHandle>("BundleReader", ctx, "DataIO", "SCIRun"),
  guiTypes_(ctx->subVar("types")),
  guiFileType_(ctx->subVar("filetype"))
{
  string importtypes = "{";
  importtypes += "{{SCIRun Bundle File} {.bdl} } ";
  importtypes += "{{SCIRun Bundle Any} {.*} } ";
  importtypes += "}";

  guiTypes_.set(importtypes);
}

BundleReader::~BundleReader(){
}

void
BundleReader::execute()
{
  const string ftpre = guiFileType_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  importing_ = false;
  GenericReader<BundleHandle>::execute();
}

void
BundleReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




