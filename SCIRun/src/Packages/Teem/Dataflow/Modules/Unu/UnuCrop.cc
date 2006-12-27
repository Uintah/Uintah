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


/*
 *  UnuCrop
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {
using namespace SCIRun;

class UnuCrop : public Module {
public:
  int valid_data(string *minS, string *maxS, int *min, int *max, Nrrd *nrrd);
  int getint(const char *str, int *n);
  UnuCrop(SCIRun::GuiContext *ctx);
  virtual ~UnuCrop();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  int parse(const NrrdDataHandle& handle, const string& val, const int axis);

private:
  vector<GuiString*> mins_;
  vector<GuiString*> maxs_;
  vector<GuiInt*> absmaxs_;
  GuiInt          num_axes_;
  GuiInt          uis_;
  GuiInt          reset_data_;
  GuiInt          digits_only_;
  vector<int>     lastmin_;
  vector<int>     lastmax_;
  int             last_generation_;
  NrrdDataHandle  last_nrrdH_;
  MatrixHandle    last_matrixH_;
};

} // End namespace SCITeem
using namespace SCITeem;
DECLARE_MAKER(UnuCrop)

UnuCrop::UnuCrop(SCIRun::GuiContext *ctx) : 
  Module("UnuCrop", ctx, Filter, "UnuAtoM", "Teem"), 
  num_axes_(get_ctx()->subVar("num-axes")),
  uis_(get_ctx()->subVar("uis")),
  reset_data_(get_ctx()->subVar("reset_data")),
  digits_only_(get_ctx()->subVar("digits_only")),
  last_generation_(-1), 
  last_nrrdH_(0)
{
  // this will get overwritten when tcl side initializes, but 
  // until then make sure it is initialized.
  num_axes_.set(0);
 
  lastmin_.resize(4, -1);
  lastmax_.resize(4, -1);  

  for (int a = 0; a < 4; a++) {
    ostringstream str;
    str << "minAxis" << a;
    mins_.push_back(new GuiString(get_ctx()->subVar(str.str())));
    ostringstream str1;
    str1 << "maxAxis" << a;
    maxs_.push_back(new GuiString(get_ctx()->subVar(str1.str())));
    ostringstream str2;
    str2 << "absmaxAxis" << a;
    absmaxs_.push_back(new GuiInt(get_ctx()->subVar(str2.str())));
  }
}


UnuCrop::~UnuCrop()
{
}


void 
UnuCrop::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrdH;
  if (!get_input_handle("Nrrd", nrrdH)) return;

  reset_vars();

  MatrixHandle matrixH;

  num_axes_.reset();

  bool new_dataset = (last_generation_ != nrrdH->generation);
  bool first_time = (last_generation_ == -1);

  // create any resample axes that might have been saved
  if (first_time) {
    uis_.reset();
    for(int i=4; i<uis_.get(); i++) {
      ostringstream str, str2, str3, str4;
      str << "minAxis" << i;
      str2 << "maxAxis" << i;
      str3 << "absmaxAxis" << i;
      str4 << i;
      mins_.push_back(new GuiString(get_ctx()->subVar(str.str())));
      maxs_.push_back(new GuiString(get_ctx()->subVar(str2.str())));
      absmaxs_.push_back(new GuiInt(get_ctx()->subVar(str3.str())));

      mins_[i]->reset();
      maxs_[i]->reset();
      lastmin_.push_back(parse(nrrdH, mins_[i]->get(),i));
      lastmax_.push_back(parse(nrrdH, maxs_[i]->get(),i));  

      get_gui()->execute(get_id().c_str() + string(" make_min_max " + str4.str()));
    }
  }

  last_generation_ = nrrdH->generation;
  num_axes_.set(nrrdH->nrrd_->dim);
  num_axes_.reset();

  // remove any unused uis or add any needes uis
  if ((unsigned  int)uis_.get() > nrrdH->nrrd_->dim) {
    // remove them
    for(unsigned int i=uis_.get()-1; i>=nrrdH->nrrd_->dim; i--) {
      ostringstream str;
      str << i;
      vector<GuiString*>::iterator iter = mins_.end();
      vector<GuiString*>::iterator iter2 = maxs_.end();
      vector<GuiInt*>::iterator iter3 = absmaxs_.end();
      vector<int>::iterator iter4 = lastmin_.end();
      vector<int>::iterator iter5 = lastmax_.end();
      mins_.erase(iter, iter);
      maxs_.erase(iter2, iter2);
      absmaxs_.erase(iter3, iter3);

      lastmin_.erase(iter4, iter4);
      lastmax_.erase(iter5, iter5);

      get_gui()->execute(get_id().c_str() + string(" clear_axis " + str.str()));
    }
    uis_.set(nrrdH->nrrd_->dim);
  } else if ((unsigned  int)uis_.get() < nrrdH->nrrd_->dim) {
    for (int i=uis_.get(); i < num_axes_.get(); i++) {
      ostringstream str, str2, str3, str4;
      str << "minAxis" << i;
      str2 << "maxAxis" << i;
      str3 << "absmaxAxis" << i;
      str4 << i;
      mins_.push_back(new GuiString(get_ctx()->subVar(str.str())));
      maxs_.push_back(new GuiString(get_ctx()->subVar(str2.str())));
      if (digits_only_.get() == 1) {
	maxs_[i]->set(to_string(nrrdH->nrrd_->axis[i].size - 1));
      }
      else {
	maxs_[i]->set("M");
      }
      absmaxs_.push_back(new GuiInt(get_ctx()->subVar(str3.str())));
      absmaxs_[i]->set(nrrdH->nrrd_->axis[i].size - 1);

      lastmin_.push_back(0);
      lastmax_.push_back(nrrdH->nrrd_->axis[i].size - 1); 

      get_gui()->execute(get_id().c_str() + string(" make_min_max " + str4.str()));
    }
    uis_.set(nrrdH->nrrd_->dim);
  }
  

  if (new_dataset) {
    for (int a=0; a<num_axes_.get(); a++) {
      int max = nrrdH->nrrd_->axis[a].size - 1;
      maxs_[a]->reset();
      absmaxs_[a]->set(nrrdH->nrrd_->axis[a].size - 1);
      absmaxs_[a]->reset();
      if (parse(nrrdH, maxs_[a]->get(),a) > max) {
	if (digits_only_.get() == 1) {
	  warning("Out of bounds, resetting axis min/max");
	  mins_[a]->set(to_string(0));
	  mins_[a]->reset();
	  maxs_[a]->set(to_string(max));
	  maxs_[a]->reset();
	  lastmin_[a] = 0;
	  lastmax_[a] = max;
	} else {
	  warning("Out of bounds, setting axis min/max to 0 to M");
	  mins_[a]->set(to_string(0));
	  mins_[a]->reset();
	  maxs_[a]->set("M");
	  maxs_[a]->reset();
	  lastmin_[a] = 0;
	  lastmax_[a] = max;
	}
      }
    }

    get_gui()->execute(get_id().c_str() + string (" update_sizes "));    
  }

  if (new_dataset && !first_time && reset_data_.get() == 1) {
    ostringstream str;
    str << get_id().c_str() << " reset_vals" << endl; 
    get_gui()->execute(str.str());  
  }
  
  if (num_axes_.get() == 0) {
    warning("Trying to crop a nrrd with no axes" );
    return;
  }

  for (int a=0; a<num_axes_.get(); a++) {
    mins_[a]->reset();
    maxs_[a]->reset();
    absmaxs_[a]->reset();
  }

  // If a matrix present use those values.
  if (get_input_handle("Current Index", matrixH, false))
  {
    if( num_axes_.get() != matrixH.get_rep()->nrows() ||
	matrixH.get_rep()->ncols() != 2 ) {
      error("Input matrix size does not match nrrd dimensions." );
      return;
    }

    for (int a=0; a<num_axes_.get(); a++) {
      int min, max;

      min = (int) matrixH.get_rep()->get(a, 0);
      max = (int) matrixH.get_rep()->get(a, 1);

      mins_[a]->set(to_string(min));
      mins_[a]->reset();
      maxs_[a]->set(to_string(max));
      mins_[a]->reset();
    }
  }

  // See if any of the sizes have changed.
  bool update = new_dataset;

  for (int i=0; i<num_axes_.get(); i++) {
      mins_[i]->reset();
    int min = parse(nrrdH, mins_[i]->get(),i);
    if (lastmin_[i] != min) {
      update = true;
      lastmin_[i] = min;
    }
    maxs_[i]->reset();
    int max = parse(nrrdH, maxs_[i]->get(),i);
    if (lastmax_[i] != max) {
      update = true;
    }
  }

  if( update ||
      !last_nrrdH_.get_rep() ||
      !last_matrixH_.get_rep())
  {
    Nrrd *nin = nrrdH->nrrd_;
    Nrrd *nout = nrrdNew();

    size_t *min = scinew size_t[num_axes_.get()];
    size_t *max = scinew size_t[num_axes_.get()];

    DenseMatrix *indexMat = scinew DenseMatrix( num_axes_.get(), 2 );
    last_matrixH_ = MatrixHandle(indexMat);

    for(int i=0; i< num_axes_.get(); i++)
    {
	mins_[i]->reset();
	maxs_[i]->reset();
	min[i] = parse(nrrdH, mins_[i]->get(),i);
	max[i] = parse(nrrdH, maxs_[i]->get(),i);

      indexMat->put(i, 0, (double) min[i]);
      indexMat->put(i, 1, (double) max[i]);
      
      if (nrrdKindSize(nin->axis[i].kind) > 1 &&
	  (min[i] != 0 || max[i] != (size_t) absmaxs_[i]->get()))
      {
	warning("Trying to crop axis " + to_string(i) +
		" which does not have a kind of nrrdKindDomain or nrrdKindUnknown");
      }
    }

    bool crop_successful = true;
    if (nrrdCrop(nout, nin, min, max)) {
      char *err = biffGetDone(NRRD);
      error(string("Trouble cropping: ") + err + "\nOutputting input Nrrd");
      remark("  Input Nrrd: nin->dim=" + to_string(nin->dim));
      free(err);
      for(unsigned int a=0; a<nin->dim; a++) {
	mins_[a]->set(to_string(0));
	maxs_[a]->set(to_string(nin->axis[a].size-1));
	lastmin_[a] = min[a];
	lastmax_[a] = max[a];
      }
      crop_successful = false;
    } else {
      for(unsigned int a=0; a<nin->dim; a++) {
	lastmin_[a] = min[a];
	lastmax_[a] = max[a];
      }
    }

    delete min;
    delete max;

    NrrdData *nrrd;
    if (crop_successful)
    {
      nrrd = scinew NrrdData(nout);
    }
    else
    {
      // TODO:  Doesn't this delete twice when the ninhandle goes away?
      nrrd = scinew NrrdData(nin);
    }

    last_nrrdH_ = NrrdDataHandle(nrrd);

    // Copy the properies, kinds, and labels.
    nrrd->copy_properties(nrrdH.get_rep());

    for( unsigned int i=0; i<nin->dim; i++ ) {
      nout->axis[i].kind  = nin->axis[i].kind;
      nout->axis[i].label = airStrdup(nin->axis[i].label);
    }

    if( (nout->axis[0].kind == nrrdKind3Vector     && nout->axis[0].size != 3) ||
	(nout->axis[0].kind == nrrdKind3DSymMatrix && nout->axis[0].size != 6) )
      nout->axis[0].kind = nrrdKindDomain;
  }

  send_output_handle("Nrrd", last_nrrdH_, true);
  send_output_handle("Selected Index", last_matrixH_, true);
}


void 
UnuCrop::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("UnuCrop needs a minor command");
    return;
  }

  if( args[1] == "add_axis" ) 
  {
      uis_.reset();
      int i = uis_.get();
      ostringstream str, str2, str3, str4;
      str << "minAxis" << i;
      str2 << "maxAxis" << i;
      str3 << "absmaxAxis" << i;
      str4 << i;
      mins_.push_back(new GuiString(get_ctx()->subVar(str.str())));
      maxs_.push_back(new GuiString(get_ctx()->subVar(str2.str())));
      absmaxs_.push_back(new GuiInt(get_ctx()->subVar(str3.str())));

      lastmin_.push_back(0);
      lastmax_.push_back(0); 

      get_gui()->execute(get_id().c_str() + string(" make_min_max " + str4.str()));

      uis_.set(uis_.get() + 1);
  }
  else if( args[1] == "remove_axis" ) 
  {
    uis_.reset();
    int i = uis_.get()-1;
    ostringstream str;
    str << i;
    vector<GuiString*>::iterator iter = mins_.end();
    vector<GuiString*>::iterator iter2 = maxs_.end();
    vector<GuiInt*>::iterator iter3 = absmaxs_.end();
    vector<int>::iterator iter4 = lastmin_.end();
    vector<int>::iterator iter5 = lastmax_.end();
    mins_.erase(iter, iter);
    maxs_.erase(iter2, iter2);
    absmaxs_.erase(iter3, iter3);
    
    lastmin_.erase(iter4, iter4);
    lastmax_.erase(iter5, iter5);
    
    get_gui()->execute(get_id().c_str() + string(" clear_axis " + str.str()));
    uis_.set(uis_.get() - 1);
  }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}

int 
UnuCrop::parse(const NrrdDataHandle& nH, const string& val, const int a) {
  // parse string val which must be in the form 
  // M, M+int, M-int, m+int, int
  
  // remove white spaces
  string new_val = "";
  for(int i = 0; i<(int)val.length(); i++) {
    if (val[i] != ' ') 
      new_val += val[i];
  }
  
  int val_length = new_val.length();
  
  bool has_base = false;
  int base = 0;
  
  char op = '+';
  
  int int_result = 0;
  int start = 0;
  
  if (val_length == 0) {
    error("Error in UnuCrop::parse String length 0.");
    return 0;
  }
  
  if (new_val[0] == 'M') {
    has_base = true;
    base = nH->nrrd_->axis[a].size - 1;
  } else if (new_val[0] == 'm') { 
    has_base = true;
    base = parse(nH, mins_[a]->get(), a);
  }
  
  if (has_base && val_length == 1) {
    return base;
  }
  
  if (has_base)  {
    start = 2;
    if (new_val[1] == '+') {
      op = '+';
    } else if (new_val[1] == '-') {
      op = '-';
    } else {
      error("Error UnuCrop::parse Must have +/- operation when using M or m with integers");
      return 0;
    }
  }

  if (!string_to_int(new_val.substr(start,val_length), int_result)) {
    error("Error UnuCrop::could not convert to integer");
    return 0;
  }

  if (has_base) {
    if (op == '+') {
      int_result = base + int_result;
    } else {
      int_result = base - int_result;
    }
  }

  return int_result;
}
