//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : RawToDenseMatrix.cc<2>
//    Author : Martin Cole
//    Date   : Tue Jan 15 09:11:42 2002


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/BioPSE/share/share.h>
#include <vector>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <stdio.h>

namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE RawToDenseMatrix : public Module {
public:
  RawToDenseMatrix(GuiContext *context);

  virtual ~RawToDenseMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString units_;
  GuiString filename_;
  GuiDouble start_;
  GuiDouble end_;
  vector<string> potfiles_;
  MatrixOPort *oport_;
  string       old_filename_;
  time_t       old_filemodification_;
  MatrixHandle handle_;
};


DECLARE_MAKER(RawToDenseMatrix)


RawToDenseMatrix::RawToDenseMatrix(GuiContext *context) : 
  Module("RawToDenseMatrix", context, Source, "DataIO", "BioPSE"),
  units_(context->subVar("units")),
  filename_(context->subVar("filename")),
  start_(context->subVar("min")),
  end_(context->subVar("max")),
  oport_(0),
  old_filename_("bogus"),
  old_filemodification_(0),
  handle_(0)
{
}

RawToDenseMatrix::~RawToDenseMatrix(){
}

void RawToDenseMatrix::execute(){
  oport_ = (MatrixOPort *)get_oport("DenseMatrix");
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
      gui->eval(id + " working_files " + filename_.get(), dummy);
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

  oport_->send(handle_);
}

void
RawToDenseMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("RawToDenseMatrix needs a minor command");
    return;
  }
  if (args[1] == "add-pot") {
    potfiles_.push_back(args[2]);
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace BioPSE


