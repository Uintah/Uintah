/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
  bool maybe_read_nrrd();
  void get_nrrd_info(NrrdDataHandle handle);

  GuiString       label_;
  GuiString       type_;
  GuiString       axis_;
  GuiString       filename_;
  NrrdDataHandle  read_handle_;
  NrrdDataHandle  send_handle_;
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
  label_(ctx->subVar("label")),
  type_(ctx->subVar("type")),
  axis_(ctx->subVar("axis")),
  filename_(ctx->subVar("filename")),
  read_handle_(0),
  send_handle_(0),
  old_filemodification_(0),
  cached_label_generation_(0),
  cached_label_(0)
{
}

NrrdReader::~NrrdReader()
{
  if (cached_label_) { delete [] cached_label_; cached_label_ = 0; }
}

void 
NrrdReader::get_nrrd_info(NrrdDataHandle handle)
{
  // Clear any old info.
  ostringstream clear; 
  clear << id.c_str() << " clear_axis_info";
  gui->execute(clear.str());
  
  if (!handle.get_rep()) { return; }
  // Send the axis info to the gui.

  const string addnew =
    id + " add_axis_info CreateNewTuple FromBelow Unknown --- --- --- ---";
  gui->execute(addnew);

  // Call the following tcl method:
  // add_axis_info {id label center size spacing min max}
  for (int i = 0; i < handle->nrrd->dim; i++) {
    ostringstream add; 
    add << id.c_str() << " add_axis_info ";
    add << i << " ";
    if (!handle->nrrd->axis[i].label) {
      handle->nrrd->axis[i].label = strdup("---");
    }
    add << handle->nrrd->axis[i].label << " ";
    switch (handle->nrrd->axis[i].center) {
    case nrrdCenterUnknown :
      add << "Unknown ";
      break;
    case nrrdCenterNode :
      add << "Node ";
      break;
    case nrrdCenterCell :
      add << "Cell ";
      break;
    }
    add << handle->nrrd->axis[i].size << " ";
    add << handle->nrrd->axis[i].spacing << " ";

    if (!(AIR_EXISTS(handle->nrrd->axis[i].min) && 
	  AIR_EXISTS(handle->nrrd->axis[i].max)))
      nrrdAxisMinMaxSet(handle->nrrd, i, nrrdCenterNode);
    
    add << handle->nrrd->axis[i].min << " ";
    add << handle->nrrd->axis[i].max << endl;  
    gui->execute(add.str());
  }
}


// Return true if handle_ was changed, otherwise return false.  This
// return value does not necessarily signal an error!
bool
NrrdReader::maybe_read_nrrd() 
{
  filename_.reset();
  string fn(filename_.get());
  if (fn == "") { return false; }

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
    }
    return true;
  }
  return false;
}



void
NrrdReader::execute()
{
  update_state(NeedData);

  if (maybe_read_nrrd())
  {
    get_nrrd_info(read_handle_);
  }

  if (!read_handle_.get_rep()) { 
    error("Please load and set up the axes for a nrrd.");
    return; 
  }

  axis_.reset();
  if (axis_.get() == "") {
    error("Please select the axis which is tuple from the UI");
    return;
  }

  // Compute which axis was picked.
  string ax(axis_.get());
  int axis = 0;
  if (ax.size()) {
    axis = atoi(ax.substr(4).c_str()); // Trim 'axis' from the string.
  }

  if (cached_label_generation_ == read_handle_->generation &&
      (cached_label_ == 0 ||
       strcmp(read_handle_->nrrd->axis[0].label, cached_label_) != 0))
  {
    if (read_handle_->nrrd->axis[0].label)
    {
      delete [] read_handle_->nrrd->axis[0].label;
      read_handle_->nrrd->axis[0].label = 0;
    }
    if (cached_label_)
    {
      read_handle_->nrrd->axis[0].label = strdup(cached_label_);
    }
  }
  
  bool added_tuple_axis = false;
  if (ax == "axisCreateNewTuple" && !added_tuple_axis)
  {
    // do add permute work here.
    Nrrd *pn = nrrdNew();
    if (nrrdAxesInsert(pn, read_handle_->nrrd, 0))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error adding a tuple axis: ") + err);
      free(err);
      return;
    }
    NrrdData *newnrrd = new NrrdData();
    newnrrd->nrrd = pn;
    newnrrd->copy_sci_data(*read_handle_.get_rep());
    send_handle_ = newnrrd;
    added_tuple_axis = true;
  }
  else if (axis != 0)
  {
    // Permute so that 0 is the tuple axis.
    const int sz = read_handle_->nrrd->dim;
    int perm[NRRD_DIM_MAX];
    Nrrd *pn = nrrdNew();
    // Init the perm array.
    for(int i = 0; i < sz; i++)
    {
      perm[i] = i;
    }

    // Swap the selected axis with 0.
    perm[0] = axis;
    perm[axis] = 0;

    if (nrrdAxesPermute(pn, read_handle_->nrrd, perm))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error adding a tuple axis: ") + err);
      free(err);
      return;
    }
    NrrdData *newnrrd = new NrrdData();
    newnrrd->nrrd = pn;
    newnrrd->copy_sci_data(*read_handle_.get_rep());
    send_handle_ = newnrrd;
  }
  else
  {
    send_handle_ = read_handle_;
  }

  // If the tuple label is valid use it. If not use the string provided
  // in the gui.
  vector<string> elems;
  if (added_tuple_axis || (! send_handle_->get_tuple_indecies(elems)))
  {
    int axis_size = send_handle_->nrrd->axis[0].size;

    // Set tuple axis name.
    label_.reset();
    type_.reset();
    string label(label_.get() + ":" + type_.get());

    string full_label = label;
    int count;
    if (type_.get() == "Scalar") count=axis_size-1;
    else if (type_.get() == "Vector") count=axis_size/3-1;
    else /* if (type_.get() == "Tensor") */ count=axis_size/7-1;
    while (count > 0) {
      full_label += string("," + label);
      count--;
    }
    // Cache off a copy of the prior label in case of axis change
    // later.
    if (send_handle_.get_rep() == read_handle_.get_rep())
    {
      if (cached_label_) { delete [] cached_label_; cached_label_ = 0; }
      if (read_handle_->nrrd->axis[0].label)
      {
	cached_label_ = strdup(read_handle_->nrrd->axis[0].label);
      }
      cached_label_generation_ = read_handle_->generation;
    }

    // TODO:  This appears to memory leak the existing label string.
    send_handle_->nrrd->axis[0].label = strdup(full_label.c_str());
  }

  // Send the data downstream.
  NrrdOPort *outport = (NrrdOPort *)get_oport("Output Data");
  if (!outport) {
    error("Unable to initialize oport 'Outport Data'.");
    return;
  }
  outport->send(send_handle_);

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
  
  if (args[1] == "read_nrrd")
  {
    if (maybe_read_nrrd())
    {
      get_nrrd_info(read_handle_);
    }
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}
