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
 *  PathReader.cc: Read a persistent camera path from a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/PathPort.h>
#include <sys/stat.h>

namespace SCIRun {

class PathReader : public Module {
  PathOPort* outport_;
  GuiString filename_;
  PathHandle handle_;
  string old_filename_;
  time_t old_filemodification_;
public:
  PathReader(const string& id);
  virtual ~PathReader();
  virtual void execute();
};

extern "C" Module* make_PathReader(const string& id) {
  return new PathReader(id);
}

PathReader::PathReader(const string& id)
  : Module("PathReader", id, Source, "DataIO", "SCIRun"),
    filename_("filename", id, this),
    old_filemodification_(0)
{
  // Create the output port
  outport_=scinew PathOPort(this, "Output Data", PathIPort::Atomic);
  add_oport(outport_);
}

PathReader::~PathReader()
{
}

void PathReader::execute()
{
  const string fn(filename_.get());

  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if (stat(fn.c_str(), &buf)) {
    error("File '" + fn + "' not found.");
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
    Piostream* stream=auto_istream(fn);
    if(!stream){
      error("Error reading file '" + fn + "'.");
      return;
    }
    
    // Read the file
    Pio(*stream, handle_);
    if(!handle_.get_rep() || stream->error()) {
      error("Error reading Path from file '" + fn + "'.");
      delete stream;
      return;
    }
    delete stream;
  }
  
  // Send the data downstream
  outport_->send(handle_);
}

} // End namespace SCIRun
