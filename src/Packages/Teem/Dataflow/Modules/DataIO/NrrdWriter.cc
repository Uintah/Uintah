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
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <sstream>
#include <fstream>
using std::ifstream;
using std::ostringstream;

namespace SCITeem {

using namespace SCIRun;

class NrrdWriter : public Module {
    NrrdIPort*  inport_;
    GuiFilename filename_;
    GuiString   filetype_;
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
  //  static int counter = 1;
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


