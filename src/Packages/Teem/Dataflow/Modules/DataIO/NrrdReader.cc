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
  void read_nrrd();
  void get_nrrd_info();

  NrrdOPort      *outport_;

  GuiString       label_;
  GuiString       type_;
  GuiString       axis_;
  GuiInt          add_;
  GuiString       filename_;
  NrrdDataHandle  handle_;
  string          old_filename_;
  time_t          old_filemodification_;
};

} // end namespace SCITeem

using namespace SCITeem;

DECLARE_MAKER(NrrdReader)

NrrdReader::NrrdReader(SCIRun::GuiContext* ctx) : 
  Module("NrrdReader", ctx, Filter, "DataIO", "Teem"),
  label_(ctx->subVar("label")),
  type_(ctx->subVar("type")),
  axis_(ctx->subVar("axis")),
  add_(ctx->subVar("add")),
  filename_(ctx->subVar("filename")),
  old_filemodification_(0)
{
}

NrrdReader::~NrrdReader()
{
}

void 
NrrdReader::get_nrrd_info()
{
  if (!handle_.get_rep()) { return; }
  // send the axis info to the gui.

  // clear any old info
  ostringstream clear; 
  clear << id.c_str() << " clear_axis_info";
  gui->execute(clear.str());
  
  // call the following tcl method
  // add_axis_info {id label center size spacing min max} 
  for (int i = 0; i < handle_->nrrd->dim; i++) {
    ostringstream add; 
    add << id.c_str() << " add_axis_info ";
    add << i << " ";
    if (!handle_->nrrd->axis[i].label) {
      handle_->nrrd->axis[i].label = strdup("---");
    }
    add << handle_->nrrd->axis[i].label << " ";
    switch (handle_->nrrd->axis[i].center) {
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
    add << handle_->nrrd->axis[i].size << " ";
    add << handle_->nrrd->axis[i].spacing << " ";

    if (!(AIR_EXISTS(handle_->nrrd->axis[i].min) && 
	  AIR_EXISTS(handle_->nrrd->axis[i].max)))
      nrrdAxisMinMaxSet(handle_->nrrd, i, nrrdCenterNode);
    
    add << handle_->nrrd->axis[i].min << " ";
    add << handle_->nrrd->axis[i].max << endl;  
    gui->execute(add.str());
  }
}

void
NrrdReader::read_nrrd() 
{
  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  filename_.reset();
  string fn(filename_.get());

  if (stat(fn.c_str(), &buf)) {
    error(string("FieldReader error - file not found ")+fn);
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif
  if(!handle_.get_rep() || 
     fn != old_filename_ || 
     new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_=fn;


    int len = fn.size();
    const string ext(".nd");

    // check that the last 3 chars are .nd for us to pio
    if (fn.substr(len - 3, 3) == ext) {
      Piostream *stream = auto_istream(fn);
      if (!stream)
      {
	error("Error reading file '" + fn + "'.");
	return;
      }

      // Read the file
      Pio(*stream, handle_);
      if (!handle_.get_rep() || stream->error())
      {
	error("Error reading data from file '" + fn +"'.");
	delete stream;
	return;
      }
      delete stream;
    } else { // assume it is just a nrrd

      NrrdData *n = scinew NrrdData;
      if (nrrdLoad(n->nrrd=nrrdNew(), strdup(fn.c_str()), 0)) {
	char *err = biffGetDone(NRRD);
	error("Read error on '" + fn + "': " + err);
	free(err);
	return;
      }
      handle_ = n;

    }
  }
}

void NrrdReader::execute()
{
  outport_ = (NrrdOPort *)get_oport("Output Data");
  if (!outport_) {
    error("Unable to initialize oport 'Outport Data'.");
    return;
  }
  filename_.reset();
  if (! handle_.get_rep() && filename_.get() != "") {
    read_nrrd();
    get_nrrd_info();
  }
  if (!handle_.get_rep()) { 
    error("Please load and set up the axes for a nrrd.");
    return; 
  }

  axis_.reset();
  add_.reset();
  if (!add_.get() && axis_.get() == "") {
    error("Please select the axis which is tuple from the UI");
    return;
  }

  if (add_.get()) {
    // do add permute work here.
    Nrrd *pn = nrrdNew();
    handle_->nrrd->axis[handle_->nrrd->dim].size = 1;
    handle_->nrrd->dim += 1;
    const int sz = handle_->nrrd->dim;
    int perm[NRRD_DIM_MAX];
    perm[0] = sz - 1; 
    for(int i = 1; i < sz; i++) {
      perm[i] = i - 1;
    }
    if (nrrdAxesPermute(pn, handle_->nrrd, perm)){
      char *err = biffGetDone(NRRD);
      error(string("Error adding a tuple axis: ") + err);
      free(err);
      return;
    }
    NrrdData *newnrrd = new NrrdData();
    newnrrd->nrrd = pn;
    newnrrd->copy_sci_data(*handle_.get_rep());
    handle_ = newnrrd;
  }

  label_.reset();
  type_.reset();
  // set tuple axis name.
  string lbl(label_.get() + ":" + type_.get());
  

  string ax(axis_.get());
  int axis = 0;
  if (ax.size()) {
    ax.erase(ax.begin(), ax.begin() + 4); // get rid of the word axis
    axis = atoi(ax.c_str());
  }

  if (axis != 0) {
    // purmute so that 0 is the tuple axis
    const int sz = handle_->nrrd->dim;
    int perm[NRRD_DIM_MAX];
    Nrrd *pn = nrrdNew();
    // init the perm array.
    for(int i = 0; i < sz; i++) {
      perm[i] = i;
    }
    //swap the selected axis with 0
    perm[0] = axis;
    perm[axis] = 0;

    if (nrrdAxesPermute(pn, handle_->nrrd, perm)){
      char *err = biffGetDone(NRRD);
      error(string("Error adding a tuple axis: ") + err);
      free(err);
      return;
    }
    NrrdData *newnrrd = new NrrdData();
    newnrrd->nrrd = pn;
    newnrrd->copy_sci_data(*handle_.get_rep());
    handle_ = newnrrd;
  }

  // if the tuple label is valid use it. If not use the string provided
  // in the gui.
  vector<string> elems;
  if (add_.get() || (! handle_->get_tuple_indecies(elems))) {
    handle_->nrrd->axis[0].label = strdup(lbl.c_str());
  }
  // Send the data downstream
  outport_->send(handle_);
}

void 
NrrdReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("NrrdReader needs a minor command");
    return;
  }
  
  if (args[1] == "read_nrrd") {
    read_nrrd();
    get_nrrd_info();
  }else {
    Module::tcl_command(args, userdata);
  }
}
