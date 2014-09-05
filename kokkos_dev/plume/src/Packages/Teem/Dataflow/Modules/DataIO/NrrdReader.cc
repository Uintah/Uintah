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
 *  NrrdReader.cc: Read in a Nrrd
 *
 *  Written by:
 *   David Weinstein
 *   School of Computing
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Util/sci_system.h>
#include <Core/Containers/StringUtil.h>
#include <Core/ImportExport/Nrrd/NrrdIEPlugin.h>
#include <sys/stat.h>
#include <sstream>
#ifdef _WIN32
#include <process.h> // for getpid
#endif

namespace SCITeem {

using namespace SCIRun;

class NrrdReader : public Module {
public:
  NrrdReader(SCIRun::GuiContext* ctx);
  virtual ~NrrdReader();
  virtual void execute();

private:
  bool read_nrrd();
  bool read_file(string filename);
  bool write_tmpfile(string filename, string *tmpfilename, string conv_command);
  bool call_importer(const string &filename);

  GuiFilename     filename_;
  GuiString       gui_types_;
  GuiString       gui_filetype_;
  
  NrrdDataHandle  read_handle_;

  string          old_filename_;
  time_t          old_filemodification_;
  int             cached_label_generation_;
  char *          cached_label_;
  bool            importing_;
};

} // end namespace SCITeem


using namespace SCITeem;

DECLARE_MAKER(NrrdReader)


NrrdReader::NrrdReader(SCIRun::GuiContext* ctx) : 
  Module("NrrdReader", ctx, Filter, "DataIO", "Teem"),
  filename_(ctx->subVar("filename")),
  gui_types_(ctx->subVar("types", false)),
  gui_filetype_(ctx->subVar("filetype")),
  read_handle_(0),
  old_filemodification_(0),
  cached_label_generation_(0),
  cached_label_(0),
  importing_(false)
{
  NrrdIEPluginManager mgr;
  vector<string> importers;
  mgr.get_importer_list(importers);
  
  string importtypes = "{";
  importtypes += "{{Nrrd Files} {.nhdr .nrrd .nd .vff .pic .pict .vol .v} } ";
  importtypes += "{{NrrdData File}     {.nd} } ";
  importtypes += "{{VFF File}          {.vff} } ";
  importtypes += "{{PICT File}         {.pic .pict} } ";
  importtypes += "{{geovoxel VOL File} {.vol} } ";
  importtypes += "{{Vista File}        {.v} } ";
  importtypes += "{{All Files}         {.*} } ";

  for (unsigned int i = 0; i < importers.size(); i++)
  {
    NrrdIEPlugin *pl = mgr.get_plugin(importers[i]);
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


NrrdReader::~NrrdReader()
{
  if (cached_label_) { delete [] cached_label_; cached_label_ = 0; }
}


// Return true if handle_ was changed, otherwise return false.  This
// return value does not necessarily signal an error!
bool
NrrdReader::read_nrrd() 
{
  filename_.reset();
  string fn(filename_.get());
  if (fn == "") { 
    error("No file has been selected.  Please choose a file.");
    return false; 
  }

  // Read the status of this file so we can compare modification timestamps.
  struct stat buf;
  if (stat(fn.c_str(), &buf) == - 1)
  {
    if (!importing_)
    {
      error("File '" + fn + "' not found.");
      return false;
    }
    else
    {
      warning("File '" + fn + "' not found.  Maybe the plugin can find it.");

      // This causes the item to cache.  Maybe a forced reread would be better?
#ifdef __sgi
      buf.st_mtim.tv_sec = 0;
#else
      buf.st_mtime = 0;
#endif
    }
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

  if (!read_handle_.get_rep() || 
      fn != old_filename_ || 
      new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_=fn;
    read_handle_ = 0;

    if (importing_)
    {
      if (!call_importer(fn))
      {
        error("Import failed.");
        return false;
      }
      return true;
    }

    int len = fn.size();
    // Filename as string
    const string filename(fn);
    const string ext(".nd");
    const string vff_ext(".vff");
    const string vff_conv_command("vff2nrrd %f %t");
    const string pic_ext(".pic");
    const string pic_ext2(".pict");
    const string pic_conv_command("PictToNrrd %f %t");
    const string vol_ext(".vol");
    const string vol_conv_command("GeoProbeToNhdr %f %t");
    const string vista_ext(".v");
    const string vista_conv_command("VistaToNrrd %f %t");

    // check that the last 3 chars are .nd for us to pio
    if (fn.substr(len - ext.size(), ext.size()) == ext)
    {
      Piostream *stream = auto_istream(fn, this);
      if (!stream)
      {
	error("Error reading file '" + fn + "'.");
	return true;
      }

      // Read the file
      Pio(*stream, read_handle_);
      if (!read_handle_.get_rep() || stream->error())
      {
	error("Error reading data from file '" + fn +"'.");
	delete stream;
	return true;
      }
      delete stream;
    }
    else
    { // assume it is just a nrrd
      if (fn.substr(len - vff_ext.size(), vff_ext.size()) == vff_ext)
      {
	string tmpfilename;
	write_tmpfile(filename, &tmpfilename, vff_conv_command);
	return read_file(tmpfilename);	
      }
      else if ((fn.substr(len - pic_ext.size(), pic_ext.size()) == pic_ext) ||
               fn.substr(len - pic_ext2.size(), pic_ext2.size()) == pic_ext2)
      {
	string tmpfilename;
	write_tmpfile(filename, &tmpfilename, pic_conv_command);
	return read_file(tmpfilename);
      }
      else if (fn.substr(len - vol_ext.size(), vol_ext.size()) == vol_ext)
      {
	string tmpfilename;
	write_tmpfile(filename, &tmpfilename, vol_conv_command);
	return read_file(tmpfilename);
      }
      else if (fn.substr(len - vista_ext.size(), vista_ext.size()) ==
               vista_ext)
      {
	string tmpfilename;
	write_tmpfile(filename, &tmpfilename, vista_conv_command);
	return read_file(tmpfilename);
      }
      else
      {
	return read_file(fn);
      }
    }
    return true;
  }
  return false;
}


bool
NrrdReader::read_file(string fn)
{
  NrrdData *n = scinew NrrdData;
  if (nrrdLoad(n->nrrd = nrrdNew(), airStrdup(fn.c_str()), 0))
  {
    char *err = biffGetDone(NRRD);
    error("Read error on '" + fn + "': " + err);
    free(err);
    return true;
  }
  read_handle_ = n;
  for (int i = 0; i < read_handle_->nrrd->dim; i++)
  {
    if (!(airExists(read_handle_->nrrd->axis[i].min) && 
	  airExists(read_handle_->nrrd->axis[i].max)) &&
	!(airExists(read_handle_->nrrd->axis[i].spaceDirection[0])))

      nrrdAxisInfoMinMaxSet(read_handle_->nrrd, i, nrrdCenterNode);
  }
  return false;
}


bool
NrrdReader::write_tmpfile(string filename, string* tmpfilename,
                          string conv_command)
{
  string::size_type loc = filename.find_last_of("/");
  const string basefilename =
    (loc==string::npos)?filename:filename.substr(loc+1);
	
  // Base filename with first extension removed.
  loc = basefilename.find_last_of(".");
  const string basenoext = basefilename.substr(0, loc);

  // Filename with first extension removed.
  loc = filename.find_last_of(".");
  const string noext = filename.substr(0, loc);

  // Temporary filename.
  //tmpfilename = "/tmp/" + basenoext + "-" +
  //  to_string((unsigned int)(getpid())) + ".nhdr";
  tmpfilename->assign("/tmp/" + basenoext + "-" +
		      to_string((unsigned int)(getpid())) + ".nhdr");
  
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    conv_command;
  while ((loc = command.find("%f")) != string::npos)
  {
    command.replace(loc, 2, "'"+filename+"'");
  }
  while ((loc = command.find("%t")) != string::npos)
  {
    command.replace(loc, 2, "'"+*tmpfilename+"'");
  }
  const int status = sci_system(command.c_str());
  if (status)
  {
    remark("'" + command + "' failed.  Read may not work.");
  }
  return true;
}


void
NrrdReader::execute()
{
  update_state(NeedData);

  const string ftpre = gui_filetype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  importing_ = !(ft == "" ||
                 ft == "Nrrd Files" ||
                 ft == "NrrdData File" ||
                 ft == "VFF File" ||
                 ft == "PICT File" ||
                 ft == "geovoxel VOL File" ||
                 ft == "Vista File" ||
                 ft == "All Files");

  read_nrrd();

  if (!read_handle_.get_rep()) { 
    error("Please load a nrrd.");
    return; 
  }

  // Send the data downstream.
  NrrdOPort *outport = (NrrdOPort *)get_oport("Output Data");
  outport->send(read_handle_);

  update_state(Completed);
}


bool
NrrdReader::call_importer(const string &filename)
{
  const string ftpre = gui_filetype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);
  
  NrrdIEPluginManager mgr;
  NrrdIEPlugin *pl = mgr.get_plugin(ft);
  if (pl)
  {
    read_handle_ = pl->fileReader_(this, filename.c_str());
    msgStream_flush();
    return read_handle_.get_rep();
  }
  return false;
}


