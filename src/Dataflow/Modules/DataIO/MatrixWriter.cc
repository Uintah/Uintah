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
#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>

namespace SCIRun {

template class GenericWriter<MatrixHandle>;

class MatrixWriter : public GenericWriter<MatrixHandle> {
protected:
  virtual bool call_exporter(const string &filename);

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


bool
MatrixWriter::call_exporter(const string &filename)
{
  MatrixIEPluginManager mgr;
  MatrixIEPlugin *pl = mgr.get_plugin("SomePlugin");
  if (pl)
  {
    return pl->filewriter(this, handle_, filename.c_str());
  }
  return false;
}


void
MatrixWriter::execute()
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

  if (exporting_)
  {
    if (!call_exporter(fn))
    {
      error("Export failed.");
      return;
    }
  }
  else
  {
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
}

} // End namespace SCIRun
