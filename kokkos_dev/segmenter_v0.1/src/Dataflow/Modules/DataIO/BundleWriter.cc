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
 *  BundleWriter.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/DataIO/GenericWriter.h>

using namespace SCIRun;
using namespace std;


class BundleWriter  : public GenericWriter<BundleHandle> {
public:
  BundleWriter(GuiContext*);

  virtual ~BundleWriter();

  virtual void execute();
protected:
  GuiString guiTypes_;
  GuiString guiFileType_;
};


DECLARE_MAKER(BundleWriter)
BundleWriter::BundleWriter(GuiContext* ctx)
  : GenericWriter<BundleHandle>("BundleWriter", ctx, "DataIO", "SCIRun"),
    guiTypes_(get_ctx()->subVar("types"), false),
    guiFileType_(get_ctx()->subVar("filetype"), "Binary")
{
  string exporttypes = "{";
  exporttypes += "{{SCIRun Bundle File} {.bdl} } ";
  exporttypes += "{{SCIRun Bundle Any} {.*} } ";
  exporttypes += "}";

  guiTypes_.set(exporttypes);
}

BundleWriter::~BundleWriter()
{
}

void
BundleWriter::execute()
{
  const string ftpre = guiFileType_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  exporting_ = false;
  GenericWriter<BundleHandle>::execute();
}

