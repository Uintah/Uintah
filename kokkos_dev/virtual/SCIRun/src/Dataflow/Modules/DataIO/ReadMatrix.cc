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


/*
 *  ReadMatrix.cc: Read a persistent matrix from a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Modules/DataIO/GenericReader.h>
#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>

namespace SCIRun {

template class GenericReader<MatrixHandle>;

class ReadMatrix : public GenericReader<MatrixHandle> {
protected:
  GuiString gui_types_;
  GuiString gui_filetype_;

  virtual bool call_importer(const string &filename);

public:
  ReadMatrix(GuiContext* ctx);
  virtual ~ReadMatrix();

  virtual void execute();
};

DECLARE_MAKER(ReadMatrix)
ReadMatrix::ReadMatrix(GuiContext* ctx)
  : GenericReader<MatrixHandle>("ReadMatrix", ctx, "DataIO", "SCIRun"),
    gui_types_(get_ctx()->subVar("types", false)),
    gui_filetype_(get_ctx()->subVar("filetype"))
{
  MatrixIEPluginManager mgr;
  vector<string> importers;
  mgr.get_importer_list(importers);
  
  string importtypes = "{";
  importtypes += "{{SCIRun Matrix File} {.mat} } ";
  importtypes += "{{SCIRun Matrix Any} {.*} } ";

  for (unsigned int i = 0; i < importers.size(); i++)
  {
    MatrixIEPlugin *pl = mgr.get_plugin(importers[i]);
    if (pl->fileExtension_ != "")
    {
      importtypes += "{{" + importers[i] + "} {" + pl->fileExtension_ + "} } ";
    }
    else
    {
      importtypes += "{{" + importers[i] + "} {.*} } ";
    }
  }

  importtypes += "}";

  gui_types_.set(importtypes);
}


ReadMatrix::~ReadMatrix()
{
}


bool
ReadMatrix::call_importer(const string &filename)
{
  const string ftpre = gui_filetype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);
  
  MatrixIEPluginManager mgr;
  MatrixIEPlugin *pl = mgr.get_plugin(ft);
  if (pl)
  {
    handle_ = pl->fileReader_(this, filename.c_str());
    msg_stream_flush();
    return handle_.get_rep();
  }
  return false;
}


void
ReadMatrix::execute()
{
  const string ftpre = gui_filetype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  importing_ = !(ft == "" ||
		 ft == "SCIRun Matrix File" ||
		 ft == "SCIRun Matrix Any");
  GenericReader<MatrixHandle>::execute();
}


} // End namespace SCIRun
