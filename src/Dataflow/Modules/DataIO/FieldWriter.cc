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
 *  FieldWriter.cc: Save persistent representation of a field to a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/DataIO/GenericWriter.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>

namespace SCIRun {

template class GenericWriter<FieldHandle>;

class FieldWriter : public GenericWriter<FieldHandle> {
protected:
  GuiString gui_types_;
  GuiString gui_exporttype_;

  virtual bool call_exporter(const string &filename);

public:
  FieldWriter(GuiContext* ctx);

  virtual void execute();
};


DECLARE_MAKER(FieldWriter)


FieldWriter::FieldWriter(GuiContext* ctx)
  : GenericWriter<FieldHandle>("FieldWriter", ctx, "DataIO", "SCIRun"),
    gui_types_(ctx->subVar("types", false)),
    gui_exporttype_(ctx->subVar("exporttype"))
{
  FieldIEPluginManager mgr;
  vector<string> exporters;
  mgr.get_exporter_list(exporters);
  
  string exporttypes = "{";
  exporttypes += "{{SCIRun Field Binary} {.fld} } ";
  exporttypes += "{{SCIRun Field ASCII} {.fld} } ";

  for (unsigned int i = 0; i < exporters.size(); i++)
  {
    FieldIEPlugin *pl = mgr.get_plugin(exporters[i]);
    if (pl->fileextension != "")
    {
      exporttypes += "{{" + exporters[i] + "} {" + pl->fileextension + "} } ";
    }
    else
    {
      exporttypes += "{{" + exporters[i] + "} {.*} } ";
    }
  }

  exporttypes += "}";

  gui_types_.set(exporttypes);
}


bool
FieldWriter::call_exporter(const string &filename)
{
  const string ftpre = gui_exporttype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);
  
  FieldIEPluginManager mgr;
  FieldIEPlugin *pl = mgr.get_plugin(ft);
  if (pl)
  {
    const bool result = pl->filewriter(this, handle_, filename.c_str());
    msgStream_flush();
    return result;
  }
  return false;
}


void
FieldWriter::execute()
{
  const string ftpre = gui_exporttype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  exporting_ = !(ft == "" ||
		 ft == "SCIRun Field Binary" ||
		 ft == "SCIRun Field ASCII");

  // Determine if we're ASCII or Binary
  string ab = "Binary";
  if (ft == "SCIRun Field ASCII") ab = "ASCII";
  filetype_.set(ab);

  GenericWriter<FieldHandle>::execute();
}


} // End namespace SCIRun
