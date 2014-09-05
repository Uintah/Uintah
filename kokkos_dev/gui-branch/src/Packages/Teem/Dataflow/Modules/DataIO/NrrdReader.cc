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

namespace SCITeem {

using namespace SCIRun;

class NrrdReader : public Module {
  NrrdOPort* outport_;
  GuiString filename_;
  NrrdDataHandle handle_;
  string old_filename_;
  time_t old_filemodification_;
public:
  NrrdReader(const string& id);
  virtual ~NrrdReader();
  virtual void execute();
};

extern "C" Module* make_NrrdReader(const string& id) {
  return new NrrdReader(id);
}

NrrdReader::NrrdReader(const string& id)
: Module("NrrdReader", id, Source, "DataIO", "Teem"), 
  filename_("filename", id, this),
  old_filemodification_(0)
{
}

NrrdReader::~NrrdReader()
{
}

void NrrdReader::execute()
{
  string fn(filename_.get());
  outport_ = (NrrdOPort *)get_oport("Output Data");
  if (!outport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  
  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
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

    NrrdData *n = scinew NrrdData;
    if (nrrdLoad(n->nrrd=nrrdNew(), strdup(fn.c_str()))) {
      char *err = biffGet(NRRD);
      cerr << "Error reading nrrd "<<fn<<": "<<err<<"\n";
      free(err);
      biffDone(NRRD);
      return;
    }
    handle_ = n;
  }

  // Send the data downstream
  outport_->send(handle_);
}

} // End namespace SCITeem
