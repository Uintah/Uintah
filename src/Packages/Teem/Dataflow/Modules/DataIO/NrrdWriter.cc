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
#include <strstream>
#include <fstream>
using std::ifstream;
using std::ostrstream;

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

  ifstream in (fn.c_str(), ios::in);
  if( in ) { //succeeded file already exists
    //hack find . insert number there
#if 0
    // The stream method was problematic (it would add extra junk in the stream
    // for no appearent reason).  Therefor, I'll write it the C way. Phooey
    // on C++. -- James Bigler
    ostrstream convert;
    // width pads the number to 4 digits
    convert.width(4);
    // This tells the stream to fill with zeros ('0').
    convert.fill('0');
    convert << counter;
    unsigned long pos = fn.find(".");
    
    //add count to file name
    
    fn.insert(pos, convert.str());
#else
    char number[10];
    sprintf(number, "%04d\0", counter);
    unsigned long pos = fn.find(".");
    fn.insert(pos, number);
#endif
    ++counter;
  }
  in.close();

  if (nrrdSave(strdup(fn.c_str()), handle->nrrd, 0)) {
    char *err = biffGetDone(NRRD);      
    error("Write error on '" + fn + "': " + err);
    free(err);
    return;
  }
}


