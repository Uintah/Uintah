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
#include <sstream>
#include <fstream>
using std::ifstream;
using std::ostringstream;

namespace SCITeem {

using namespace SCIRun;

class NrrdWriter : public Module {
    NrrdIPort* inport_;
    GuiString filename_;
    GuiString filetype_;
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
  filetype_(ctx->subVar("filetype"))
{
}

NrrdWriter::~NrrdWriter()
{
}

void NrrdWriter::execute()
{
  static int counter = 1;
  // Read data from the input port
  NrrdDataHandle handle;
  inport_ = (NrrdIPort *)get_iport("Input Data");
  if (!inport_) {
    error("Unable to initialize iport 'Input Data'.");
    return;
  }

  if(!inport_->get(handle))
    return;
  
  if (!handle.get_rep()) {
    error("Null input");
    return;
  }

  // If no name is provided, return
  string fn(filename_.get());
  if(fn == "") {
    error("Warning: no filename in NrrdWriter");
    return;
  }

  // Open up the output stream
  Piostream* stream;
  string ft(filetype_.get());
  if (ft == "Binary")
  {
    stream = scinew BinaryPiostream(fn, Piostream::Write);
  }
  else
  {
    stream = scinew TextPiostream(fn, Piostream::Write);
  }
    
  if (stream->error()) {
    error("Could not open file for writing" + fn);
  } else {
    // Write the file
    Pio(*stream, handle); // wlll also write out a separate nrrd.
    delete stream; 
  } 
}


