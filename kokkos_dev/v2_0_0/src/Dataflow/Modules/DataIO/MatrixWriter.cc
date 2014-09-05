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
 *  MatrixWriter.cc: Save persistent representation of a matrix to a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Modules/DataIO/GenericWriter.h>

namespace SCIRun {

template class GenericWriter<MatrixHandle>;

class MatrixWriter : public GenericWriter<MatrixHandle> {
public:
  GuiInt split_;
  MatrixWriter(GuiContext* ctx);
  virtual void execute();
};


DECLARE_MAKER(MatrixWriter)

MatrixWriter::MatrixWriter(GuiContext* ctx)
  : GenericWriter<MatrixHandle>("MatrixWriter", ctx, "DataIO", "SCIRun"),
    split_(ctx->subVar("split"))
{
}

void MatrixWriter::execute()
{
  // Read data from the input port
  SimpleIPort<MatrixHandle> *inport = 
    (SimpleIPort<MatrixHandle> *)get_iport("Input Data");

  MatrixHandle handle;
  if(!inport->get(handle) || !handle.get_rep())
  {
    remark("No data on input port.");
    return;
  }

  // If no name is provided, return.
  const string fn(filename_.get());
  if (fn == "")
  {
    warning("No filename specified.");
    return;
  }

  if (!overwrite()) return;

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

  // Check whether the file should be split into header and data
  handle->set_raw(split_.get());
  
  if (stream->error()) {
    error("Could not open file for writing" + fn);
  } else {
    // Write the file
    Pio(*stream, handle);
    delete stream;
  } 
}

} // End namespace SCIRun
