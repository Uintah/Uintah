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
 *  ColorMapReader.cc: Read a persistent colormap from a file
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
 *   Uses .tcl file with "filename"
 *   Output port must be of type SimpleOPort
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <sys/stat.h>

namespace SCIRun {


template <class HType> 
class GenericReader : public Module
{
protected:
  GuiFilename filename_;
  HType     handle_;
  string    old_filename_;
  time_t    old_filemodification_;

  bool importing_;
  virtual bool call_importer(const string &filename);

public:
  GenericReader(const string &name, GuiContext* ctx,
		const string &category, const string &package);
  virtual ~GenericReader();

  virtual void execute();
};


template <class HType>
GenericReader<HType>::GenericReader(const string &name, GuiContext* ctx,
				    const string &cat, const string &pack)
  : Module(name, ctx, Source, cat, pack),
    filename_(ctx->subVar("filename")),
    old_filemodification_(0),
    importing_(false)
{
}

template <class HType>
GenericReader<HType>::~GenericReader()
{
}


template <class HType>
bool
GenericReader<HType>::call_importer(const string &/*filename*/)
{
  return false;
}


template <class HType>
void
GenericReader<HType>::execute()
{
  const string fn(filename_.get());

  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if( fn == "" ) {
    error("No file has been selected.  Please choose a file.");
    return;
  } else if (stat(fn.c_str(), &buf)) {
    if (!importing_)
    {
      error("File '" + fn + "' not found.");
      return;
    }
    else
    {
      warning("File '" + fn + "' not found.  Maybe the plugin can find it.");

      // This causes the item to cache.  Maybe a forced reread would be better?
#ifdef __sgi
      buf.st_mtim.tv_sec = 0;
#else
      buf.st_mtime = 0;
#endif
    }
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif
  if (!handle_.get_rep() || 
      fn != old_filename_ || 
      new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_ = fn;

    if (importing_)
    {
      if (!call_importer(fn))
      {
	error("Import failed.");
	return;
      }
    }
    else
    {
      Piostream *stream = auto_istream(fn, this);
      if (!stream)
      {
	error("Error reading file '" + fn + "'.");
	return;
      }
    
      // Read the file
      Pio(*stream, handle_);
      if (!handle_.get_rep() || stream->error())
      {
	error("Error reading data from file '" + fn +"'.");
	delete stream;
	return;
      }
      delete stream;
    }
  }

  // Send the data downstream.
  SimpleOPort<HType> *outport = (SimpleOPort<HType> *)getOPort(0);
  if (!outport) {
    error("Unable to initialize oport.");
    return;
  }
  outport->send(handle_);
}

} // End namespace SCIRun
