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
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

namespace SCITeem {

using namespace SCIRun;

class NrrdWriter : public Module {
    NrrdIPort* inport_;
    GuiString filename_;
    GuiString filetype_;
public:
    NrrdWriter(const string& id);
    virtual ~NrrdWriter();
    virtual void execute();
};

extern "C" Module* make_NrrdWriter(const string& id) {
  return new NrrdWriter(id);
}

NrrdWriter::NrrdWriter(const string& id)
: Module("NrrdWriter", id, Source, "DataIO", "Teem"), 
  filename_("filename", id, this),
  filetype_("filetype", id, this)
{
}

NrrdWriter::~NrrdWriter()
{
}

void NrrdWriter::execute()
{
  // Read data from the input port
  NrrdDataHandle handle;
  inport_ = (NrrdIPort *)get_iport("Input Data");
  if (!inport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }

  if(!inport_->get(handle))
    return;

  // If no name is provided, return
  string fn(filename_.get());
  if(fn == "") {
    error("Warning: no filename in NrrdWriter");
    return;
  }

  if (nrrdSave(strdup(fn.c_str()), handle->nrrd, 0)) {
    char *err = biffGet(NRRD);      
    cerr << "Error writing nrrd " << fn << ": " << err << "\n";
    free(err);
    biffDone(NRRD);
    return;
  }
}

} // End namespace SCITeem

