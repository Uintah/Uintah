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
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <sys/stat.h>
#include <sstream>

namespace SCITeem {

using namespace SCIRun;

class NrrdReader : public Module {
public:
  NrrdReader(SCIRun::GuiContext* ctx);
  virtual ~NrrdReader();
  virtual void execute();
  virtual void tcl_command(GuiArgs& args, void* userdata);
private:
  bool read_nrrd();

  GuiFilename     filename_;

  NrrdDataHandle  read_handle_;

  string          old_filename_;
  time_t          old_filemodification_;
  int             cached_label_generation_;
  char *          cached_label_;
};

} // end namespace SCITeem

using namespace SCITeem;

DECLARE_MAKER(NrrdReader)

NrrdReader::NrrdReader(SCIRun::GuiContext* ctx) : 
  Module("NrrdReader", ctx, Filter, "DataIO", "Teem"),
  filename_(ctx->subVar("filename")),
  read_handle_(0),
  old_filemodification_(0),
  cached_label_generation_(0),
  cached_label_(0)
{
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
    error("Please specify nrrd filename");
    return false; 
  }

  // Read the status of this file so we can compare modification timestamps.
  struct stat buf;
  if (stat(fn.c_str(), &buf)) {
    error(string("FieldReader error - file not found: '")+fn+"'");
    return false;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif
    if(!read_handle_.get_rep() || 
  fn != old_filename_ || 
  new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_=fn;
    read_handle_ = 0;

    int len = fn.size();
    const string ext(".nd");

    // check that the last 3 chars are .nd for us to pio
    if (fn.substr(len - 3, 3) == ext) {
      Piostream *stream = auto_istream(fn);
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
    } else { // assume it is just a nrrd

      NrrdData *n = scinew NrrdData;
      if (nrrdLoad(n->nrrd=nrrdNew(), strdup(fn.c_str()), 0)) {
	char *err = biffGetDone(NRRD);
	error("Read error on '" + fn + "': " + err);
	free(err);
	return true;
      }
      read_handle_ = n;
      for (int i = 0; i < read_handle_->nrrd->dim; i++) {
	if (!(airExists_d(read_handle_->nrrd->axis[i].min) && 
	      airExists_d(read_handle_->nrrd->axis[i].max)))
	  nrrdAxisInfoMinMaxSet(read_handle_->nrrd, i, nrrdCenterNode);
      }
    }
    return true;
    }
     return false;
}



void
NrrdReader::execute()
{
  update_state(NeedData);

  read_nrrd();

  if (!read_handle_.get_rep()) { 
    error("Please load a nrrd.");
    return; 
  }

  // Send the data downstream.
  NrrdOPort *outport = (NrrdOPort *)get_oport("Output Data");
  if (!outport) {
    error("Unable to initialize oport 'Outport Data'.");
    return;
  }

  outport->send(read_handle_);

  update_state(Completed);
}


void 
NrrdReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("NrrdReader needs a minor command");
    return;
  }
  Module::tcl_command(args, userdata);
}
