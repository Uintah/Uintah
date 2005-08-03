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
 *  ColorMapWriter.cc: Save persistent representation of a colormap to a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


/*
 * Limitations:
 *   Uses .tcl file with "filename" and "filetype"
 *   Input port must be of type SimpleIPort
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Core/Persistent/Pstreams.h>

namespace SCIRun {


template <class HType>
class GenericWriter : public Module {
protected:
  HType       handle_;
  GuiFilename filename_;
  GuiString   filetype_;
  GuiInt      confirm_;
  bool        exporting_;

  virtual bool overwrite();
  virtual bool call_exporter(const string &filename);

public:
  GenericWriter(const string &name, GuiContext* ctx,
		const string &category, const string &package);
  virtual ~GenericWriter();

  virtual void execute();
};


template <class HType>
GenericWriter<HType>::GenericWriter(const string &name, GuiContext* ctx,
				    const string &cat, const string &pack)
  : Module(name, ctx, Sink, cat, pack),
    filename_(ctx->subVar("filename")),
    filetype_(ctx->subVar("filetype")),
    confirm_(ctx->subVar("confirm")),
    exporting_(false)
{
}


template <class HType>
GenericWriter<HType>::~GenericWriter()
{
}

template <class HType>
bool
GenericWriter<HType>::overwrite()
{
  std::string result;
  gui->lock();
  gui->eval(id+" overwrite",result);
  gui->unlock();
  if (result == std::string("0")) {
    warning("User chose to not save.");
    return 0;
  }
  return 1;
}
  

template <class HType>
bool
GenericWriter<HType>::call_exporter(const string &/*filename*/)
{
  return false;
}


template <class HType>
void
GenericWriter<HType>::execute()
{
  SimpleIPort<HType> *inport = (SimpleIPort<HType> *)getIPort(0);
  if (!inport) {
    error("Unable to initialize iport.");
    return;
  }

  // Read data from the input port
  if (!inport->get(handle_) || !handle_.get_rep())
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
      stream = auto_ostream(fn, "Binary", this);
    }
    else
    {
      stream = auto_ostream(fn, "Text", this);
    }

    if (stream->error())
    {
      error("Could not open file for writing" + fn);
    }
    else
    {
      // Write the file
      Pio(*stream, handle_);
    } 
    delete stream;
  }
}

} // End namespace SCIRun
