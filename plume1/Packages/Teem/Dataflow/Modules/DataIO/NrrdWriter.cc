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
 *  NrrdWriter.cc: Nrrd writer
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/ImportExport/Nrrd/NrrdIEPlugin.h>

#include <sstream>
#include <fstream>

using std::ifstream;
using std::ostringstream;


namespace SCITeem {

using namespace SCIRun;

class NrrdWriter : public Module
{
  NrrdIPort*  inport_;
  GuiFilename filename_;
  GuiString   filetype_;
  GuiString   gui_types_;
  GuiString   gui_exporttype_;

  bool overwrite();
  bool call_exporter(const string &filename, NrrdDataHandle handle);

public:
    NrrdWriter(GuiContext *ctx);
    virtual ~NrrdWriter();
    virtual void execute();
};

} // end namespace SCITeem


using namespace SCITeem;

DECLARE_MAKER(NrrdWriter)


NrrdWriter::NrrdWriter(GuiContext *ctx)
: Module("NrrdWriter", ctx, Filter, "DataIO", "Teem"), 
  filename_(ctx->subVar("filename")),
  filetype_(ctx->subVar("filetype")),
  gui_types_(ctx->subVar("types", false)),
  gui_exporttype_(ctx->subVar("exporttype"))
{
  NrrdIEPluginManager mgr;
  vector<string> exporters;
  mgr.get_exporter_list(exporters);
  
  string exporttypes = "{";
  exporttypes += "{{Nrrd} {.nrrd} } ";
  exporttypes += "{{Nrrd Header and Raw} {.nhdr *.raw} } ";

  for (unsigned int i = 0; i < exporters.size(); i++)
  {
    NrrdIEPlugin *pl = mgr.get_plugin(exporters[i]);
    if (pl->fileExtension_ != "")
    {
      exporttypes += "{{" + exporters[i] + "} {" + pl->fileExtension_ + "} } ";
    }
    else
    {
      exporttypes += "{{" + exporters[i] + "} {.*} } ";
    }
  }

  exporttypes += "}";

  gui_types_.set(exporttypes);
}


NrrdWriter::~NrrdWriter()
{
}


bool
NrrdWriter::call_exporter(const string &filename, NrrdDataHandle handle)
{
  const string ftpre = gui_exporttype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);
  
  NrrdIEPluginManager mgr;
  NrrdIEPlugin *pl = mgr.get_plugin(ft);
  if (pl)
  {
    return pl->fileWriter_(this, handle, filename.c_str());
  }
  return false;
}


bool
NrrdWriter::overwrite()
{
  std::string result;
  gui->lock();
  gui->eval(id + " overwrite", result);
  gui->unlock();
  if (result == std::string("0"))
  {
    warning("User chose to not save.");
    return false;
  }
  return true;
}


void
NrrdWriter::execute()
{
  const string ftpre = gui_exporttype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  const bool exporting_ = !(ft == "" ||
                            ft == "Nrrd" ||
                            ft == "Nrrd Header and Raw");

  // Read data from the input port.
  NrrdDataHandle handle_;

  NrrdIPort *inport = (NrrdIPort *)get_iport("Input Data");

  // Read data from the input port
  if (!inport->get(handle_) || !handle_.get_rep())
  {
    remark("No data on input port.");
    return;
  }

  // If no name is provided, return.
  const string fn(filename_.get());
  if (fn == "")
  {
    warning("No filename specified.");
    return;
  }

  if (!overwrite()) return;
  
  if (exporting_)
  {
    if (!call_exporter(fn, handle_))
    {
      error("Export failed.");
    }
    return;
  }

  // Get root of filename (no extension).
  string::size_type e = fn.find_last_of(".");
  string root = fn;
  if (e != string::npos) root = fn.substr(0,e);

  // Determine which case we are writing out based on the
  // original filename.  
  bool writing_nrrd = false;
  bool writing_nhdr = false;
  bool writing_nd = false;

  if (fn.find(".nrrd",0) != string::npos) writing_nrrd = true;
  else if (fn.find(".nhdr",0) != string::npos) writing_nhdr = true;
  else if (fn.find(".nd",0) != string::npos) writing_nd = true;

  // If the filename doesn't have an extension
  // use the export type to determine what it should be.
  if (!writing_nrrd && !writing_nhdr && !writing_nd)
  {
    if (ft == "Nrrd") { writing_nrrd = true; }
    else { writing_nhdr = true; }
  }

  // Only write out the .nd extension if that was the filename
  // specified or if there are properties.  In the case that it
  // was an .nd filename specified, write out a .nrrd file also.
  if (handle_->nproperties() > 0)
  {
    if (!writing_nd)
    {
      writing_nd = true;
      if (!writing_nhdr)
        writing_nrrd = true;
    }
  }

  // Writing out NrrdData - use Piostream.
  if (writing_nd)
  {
    string nrrd_data_fn = root + ".nd";
    Piostream* stream;
    string ft(filetype_.get());

    // Set NrrdData's write_nrrd variable to indicate
    // whether NrrdData's io method should write out a .nrrd or .nhdr
    if (writing_nhdr) handle_.get_rep()->write_nrrd_ = false;
    else handle_.get_rep()->write_nrrd_ = true;

    if (ft == "Binary")
    {
      stream = scinew BinaryPiostream(nrrd_data_fn, Piostream::Write);
    }
    else
    {
      stream = scinew TextPiostream(nrrd_data_fn, Piostream::Write);
    }
    
    if (stream->error()) {
      error("Could not open file for writing" + nrrd_data_fn);
    } else {
      // Write the file
      Pio(*stream, handle_); // This also writes a separate nrrd.
      delete stream; 
    } 
  }
  else
  {
    // Writing out ordinary .nrrd .nhdr file
    string nrrd_fn = root;
    if (writing_nhdr) nrrd_fn += ".nhdr";
    else nrrd_fn += ".nrrd";

    Nrrd *nin = handle_->nrrd;
    
    NrrdIoState *nio = nrrdIoStateNew();
    // Set encoding to be raw
    nio->encoding = nrrdEncodingArray[1];
    // Set format to be nrrd
    nio->format = nrrdFormatArray[1];
    // Set endian to be endian of machine
    nio->endian = airMyEndian;
    
    if (AIR_ENDIAN != nio->endian)
    {
      nrrdSwapEndian(nin);
    }
    if (writing_nhdr)
    {
      if (nio->format != nrrdFormatNRRD)
      {
        nio->format = nrrdFormatNRRD;
      }
    }
    
    if (nrrdSave(nrrd_fn.c_str(), nin, nio))
    {
      char *err = biffGet(NRRD);      
      cerr << "Error writing nrrd " << nrrd_fn << ": "<< err << endl;
      free(err);
      biffDone(NRRD);
      return;
    }
  }
}


