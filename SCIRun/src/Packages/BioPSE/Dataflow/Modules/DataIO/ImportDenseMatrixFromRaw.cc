/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

//    File   : ImportDenseMatrixFromRaw.cc<2>
//    Author : Martin Cole
//    Date   : Tue Jan 15 09:11:42 2002


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <vector>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <stdio.h>

namespace BioPSE {

using namespace SCIRun;

class ImportDenseMatrixFromRaw : public Module {
public:
  ImportDenseMatrixFromRaw(GuiContext *context);

  virtual ~ImportDenseMatrixFromRaw();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString units_;
  GuiString filename_;
  GuiDouble start_;
  GuiDouble end_;
  vector<string> potfiles_;
  string       old_filename_;
  time_t       old_filemodification_;
  MatrixHandle handle_;
};


DECLARE_MAKER(ImportDenseMatrixFromRaw)


ImportDenseMatrixFromRaw::ImportDenseMatrixFromRaw(GuiContext *context) : 
  Module("ImportDenseMatrixFromRaw", context, Source, "DataIO", "BioPSE"),
  units_(context->subVar("units")),
  filename_(context->subVar("filename")),
  start_(context->subVar("min")),
  end_(context->subVar("max")),
  old_filename_("bogus"),
  old_filemodification_(0),
  handle_(0)
{
}

ImportDenseMatrixFromRaw::~ImportDenseMatrixFromRaw()
{
}


void
ImportDenseMatrixFromRaw::execute()
{
  filename_.reset();

  struct stat buf;
  if (stat(filename_.get().c_str(), &buf)) {
    error("File '" + filename_.get() + "' not found.");
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

  if (!handle_.get_rep() || 
      filename_.get() != old_filename_ || 
      new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_ = filename_.get();
    int rows = 0;
    // Parse pnt file to see how many we have.
    FILE *f = fopen(filename_.get().c_str(), "rt");
    if (f) {
      float x,y,z;
      while (!feof(f) && fscanf(f, "%f %f %f", &x, &y, &z) == 3) {
	++rows;
      }
    } else {
      error("Could not open file: " + filename_.get());
      return;
    } 
    fclose(f);

    if (potfiles_.size() == 0) {
      //then the filename was saved in a net, not entered into the gui.
      //trigger the list of file names to parse.
      string dummy;
      get_gui()->eval(get_id() + " working_files " + filename_.get(), dummy);
    }

    DenseMatrix* mat = scinew DenseMatrix(rows, potfiles_.size());
    int col = 0;
    vector<string>::iterator iter = potfiles_.begin();
    while (iter != potfiles_.end()) {
      int row = 0;
      FILE *f=fopen((*iter).c_str(), "rt");
      if (f) {
	double val;
	while(!feof(f) && fscanf(f, "%lf", &val) == 1) {
	  mat->put(row, col, val);
	  ++row;
	}
      } else {
	error("Could not open file: " + *iter);
	return;
      }
      ++col;
      ++iter;
    }
    handle_ = mat;
  }
  // Add Properties
  units_.reset();
  start_.reset();
  end_.reset();
  
  handle_->set_property("time-units", units_.get(), false);
  handle_->set_property("time-start", start_.get(), true);
  handle_->set_property("time-end", end_.get(), true);

  send_output_handle("DenseMatrix", handle_, true);
}


void
ImportDenseMatrixFromRaw::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("ImportDenseMatrixFromRaw needs a minor command");
    return;
  }
  if (args[1] == "add-pot") {
    potfiles_.push_back(args[2]);
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace BioPSE


