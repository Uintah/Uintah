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
 *  FieldReader.cc: Read a persistent field from a file
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
#include <Dataflow/Modules/DataIO/GenericReader.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>

namespace SCIRun {

template class GenericReader<FieldHandle>;

class FieldReader : public GenericReader<FieldHandle> {
protected:
  GuiString gui_types_;
  GuiString gui_filetype_;

  virtual bool call_importer(const string &filename);

public:
  FieldReader(GuiContext* ctx);

  virtual void execute();
};


DECLARE_MAKER(FieldReader)

FieldReader::FieldReader(GuiContext* ctx)
  : GenericReader<FieldHandle>("FieldReader", ctx, "DataIO", "SCIRun"),
    gui_types_(ctx->subVar("types", false)),
    gui_filetype_(ctx->subVar("filetype"))
{
  FieldIEPluginManager mgr;
  vector<string> importers;
  mgr.get_importer_list(importers);
  
  string importtypes = "{";
  importtypes += "{{SCIRun Field File} {.fld} } ";
  importtypes += "{{SCIRun Field Any} {.*} } ";

  for (unsigned int i = 0; i < importers.size(); i++)
  {
    FieldIEPlugin *pl = mgr.get_plugin(importers[i]);
    if (pl->fileextension != "")
    {
      importtypes += "{{" + importers[i] + "} {" + pl->fileextension + "} } ";
    }
    else
    {
      importtypes += "{{" + importers[i] + "} {.*} } ";
    }
  }

  importtypes += "}";

  gui_types_.set(importtypes);
}


bool
FieldReader::call_importer(const string &filename)
{
  const string ftpre = gui_filetype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);
  
  FieldIEPluginManager mgr;
  FieldIEPlugin *pl = mgr.get_plugin(ft);
  if (pl)
  {
    handle_ = pl->filereader(this, filename.c_str());
    msgStream_flush();
    return handle_.get_rep();
  }
  return false;
}


void
FieldReader::execute()
{
  const string ftpre = gui_filetype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  importing_ = !(ft == "" ||
		 ft == "SCIRun Field File" ||
		 ft == "SCIRun Field Any");
  GenericReader<FieldHandle>::execute();
}


} // End namespace SCIRun
