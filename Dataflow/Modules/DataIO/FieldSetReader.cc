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
 *  FieldSetReader.cc: Read a persistent field from a file
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldSetPort.h>
#include <sys/stat.h>

namespace SCIRun {

class FieldSetReader : public Module {
  GuiString filename_;
  FieldSetHandle handle_;
  string old_filename_;
  time_t old_filemodification_;
public:
  FieldSetReader(const string& id);
  virtual ~FieldSetReader();
  virtual void execute();
};

extern "C" Module* make_FieldSetReader(const string& id) {
  return new FieldSetReader(id);
}

FieldSetReader::FieldSetReader(const string& id)
  : Module("FieldSetReader", id, Source, "DataIO", "SCIRun"),
    filename_("filename", id, this),
    old_filemodification_(0)
{
}

FieldSetReader::~FieldSetReader()
{
}

void FieldSetReader::execute()
{
  string fn(filename_.get());

  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if (stat(fn.c_str(), &buf)) {
    error("FieldSetReader error - file not found " + fn);
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
      error("Error reading file: " + fn);
      return;
    }
    
    // Read the file
    Pio(*stream, handle_);
    if(stream->error()){
      error("Stream Error reading FieldSet from file:" + fn);
      delete stream;
      return;
    }

    if(! handle_.get_rep()){
      error("Error building FieldSet from stream: " + fn);
      delete stream;
      return;
    }
    delete stream;
  }
  
  // Send the data downstream
  FieldSetOPort *outport = (FieldSetOPort *)get_oport(0);
  outport->send(handle_);
}

} // End namespace SCIRun
