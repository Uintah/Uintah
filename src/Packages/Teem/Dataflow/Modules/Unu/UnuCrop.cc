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
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

#include <Dataflow/Ports/MatrixPort.h>
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

private:
  vector<GuiInt*> mins_;
  vector<GuiInt*> maxs_;
  vector<GuiInt*> absmaxs_;
  GuiInt          num_axes_;
  GuiInt          uis_;
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
  num_axes_(ctx->subVar("num-axes")),
  uis_(ctx->subVar("uis")),
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
    mins_.push_back(new GuiInt(ctx->subVar(str.str())));
    ostringstream str1;
    str1 << "maxAxis" << a;
    maxs_.push_back(new GuiInt(ctx->subVar(str1.str())));
    ostringstream str2;
    str2 << "absmaxAxis" << a;
    absmaxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
  }
}

UnuCrop::~UnuCrop() {
}

void 
UnuCrop::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrdH;
  NrrdIPort* inrrd = (NrrdIPort *)get_iport("Nrrd");

  if (!inrrd) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }

  if (!inrrd->get(nrrdH) || !nrrdH.get_rep()) {
    error( "No handle or representation" );
    return;
  }

  MatrixHandle matrixH;
  MatrixIPort* imatrix = (MatrixIPort *)get_iport("Current Index");

  if (!imatrix) {
    error("Unable to initialize iport 'Current Index'.");
    return;
  }

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
      mins_.push_back(new GuiInt(ctx->subVar(str.str())));
      maxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
      absmaxs_.push_back(new GuiInt(ctx->subVar(str3.str())));

      lastmin_.push_back(mins_[i]->get());
      lastmax_.push_back(maxs_[i]->get());  

      gui->execute(id.c_str() + string(" make_min_max " + str4.str()));
    }
  }

  last_generation_ = nrrdH->generation;
  num_axes_.set(nrrdH->nrrd->dim);
  num_axes_.reset();

  // remove any unused uis or add any needes uis
  if (uis_.get() > nrrdH->nrrd->dim) {
    // remove them
    for(int i=uis_.get()-1; i>=nrrdH->nrrd->dim; i--) {
      ostringstream str;
      str << i;
      vector<GuiInt*>::iterator iter = mins_.end();
      vector<GuiInt*>::iterator iter2 = maxs_.end();
      vector<GuiInt*>::iterator iter3 = absmaxs_.end();
      vector<int>::iterator iter4 = lastmin_.end();
      vector<int>::iterator iter5 = lastmax_.end();
      mins_.erase(iter, iter);
      maxs_.erase(iter2, iter2);
      absmaxs_.erase(iter3, iter3);

      lastmin_.erase(iter4, iter4);
      lastmax_.erase(iter5, iter5);

      gui->execute(id.c_str() + string(" clear_axis " + str.str()));
    }
    uis_.set(nrrdH->nrrd->dim);
  } else if (uis_.get() < nrrdH->nrrd->dim) {
    for (int i=uis_.get(); i < num_axes_.get(); i++) {
      ostringstream str, str2, str3, str4;
      str << "minAxis" << i;
      str2 << "maxAxis" << i;
      str3 << "absmaxAxis" << i;
      str4 << i;
      mins_.push_back(new GuiInt(ctx->subVar(str.str())));
      maxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
      maxs_[i]->set(nrrdH->nrrd->axis[i].size - 1);
      absmaxs_.push_back(new GuiInt(ctx->subVar(str3.str())));
      absmaxs_[i]->set(nrrdH->nrrd->axis[i].size - 1);

      lastmin_.push_back(0);
      lastmax_.push_back(nrrdH->nrrd->axis[i].size - 1); 

      gui->execute(id.c_str() + string(" make_min_max " + str4.str()));
    }
    uis_.set(nrrdH->nrrd->dim);
  }
  

  if (new_dataset) {
    for (int a=0; a<num_axes_.get(); a++) {
      int max = nrrdH->nrrd->axis[a].size - 1;
      maxs_[a]->reset();
      absmaxs_[a]->set(nrrdH->nrrd->axis[a].size - 1);
      absmaxs_[a]->reset();
      if (maxs_[a]->get() > max) {
	maxs_[a]->set(nrrdH->nrrd->axis[a].size - 1);
	maxs_[a]->reset();
      }
    }
    
    ostringstream str;
    str << id.c_str() << " set_max_vals" << endl; 
    gui->execute(str.str());  
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
  if (imatrix->get(matrixH) && matrixH.get_rep()) {

    if( num_axes_.get() != matrixH.get_rep()->nrows() ||
	matrixH.get_rep()->ncols() != 2 ) {
      error("Input matrix size does not match nrrd dimensions." );
      return;
    }

    for (int a=0; a<num_axes_.get(); a++) {
      int min, max;

      min = (int) matrixH.get_rep()->get(a, 0);
      max = (int) matrixH.get_rep()->get(a, 1);

      mins_[a]->set(min);
      mins_[a]->reset();
      maxs_[a]->set(max);
      mins_[a]->reset();
    }
  }

  // See if any of the sizes have changed.
  bool update = new_dataset;

  for (int i=0; i<num_axes_.get(); i++) {
    if (lastmin_[i] != mins_[i]->get()) {
      update = true;
      lastmin_[i] = mins_[i]->get();
    }
    if (lastmax_[i] != maxs_[i]->get()) {
      update = true;
	lastmax_[i] = maxs_[i]->get();
    }
  }

  if( update ) {
    Nrrd *nin = nrrdH->nrrd;
    Nrrd *nout = nrrdNew();

    int *min = scinew int[num_axes_.get()];
    int *max = scinew int[num_axes_.get()];

    DenseMatrix *indexMat = scinew DenseMatrix( num_axes_.get(), 2 );
    last_matrixH_ = MatrixHandle(indexMat);

    for(int i=0; i< num_axes_.get(); i++) {
      min[i] = mins_[i]->get();
      max[i] = maxs_[i]->get();

      indexMat->put(i, 0, (double) min[i]);
      indexMat->put(i, 1, (double) max[i]);

      if (nrrdKindSize(nin->axis[i].kind) > 1 &&
	  (min[i] != 0 || max[i] != absmaxs_[i]->get())) {
	warning("Trying to crop axis " + to_string(i) +
		" which does not have a kind of nrrdKindDomain or nrrdKindUnknown");
      }
    }

    if (nrrdCrop(nout, nin, min, max)) {
      char *err = biffGetDone(NRRD);
      error(string("Trouble cropping: ") + err);
      msgStream_ << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
      free(err);
    }

    delete min;
    delete max;

    NrrdData *nrrd = scinew NrrdData;
    nrrd->nrrd = nout;
    last_nrrdH_ = NrrdDataHandle(nrrd);

    // Copy the properies, kinds, and labels.
    *((PropertyManager *)nrrd) = *((PropertyManager *)(nrrdH.get_rep()));

    for( int i=0; i<nin->dim; i++ ) {
      nout->axis[i].kind  = nin->axis[i].kind;
      nout->axis[i].label = nin->axis[i].label;
    }

    if( (nout->axis[0].kind == nrrdKind3Vector     && nout->axis[0].size != 3) ||
	(nout->axis[0].kind == nrrdKind3DSymTensor && nout->axis[0].size != 6) )
      nout->axis[0].kind = nrrdKindDomain;
  }

  if (last_nrrdH_.get_rep()) {

    NrrdOPort* onrrd = (NrrdOPort *)get_oport("Nrrd");
    if (!onrrd) {
      error("Unable to initialize oport 'Nrrd'.");
      return;
    }

    onrrd->send(last_nrrdH_);
  }

  if (last_matrixH_.get_rep()) {
    MatrixOPort* omatrix = (MatrixOPort *)get_oport("Selected Index");
    
    if (!omatrix) {
      error("Unable to initialize oport 'Selected Index'.");
      return;
    }
    
    omatrix->send( last_matrixH_ );
  }
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
      mins_.push_back(new GuiInt(ctx->subVar(str.str())));
      maxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
      absmaxs_.push_back(new GuiInt(ctx->subVar(str3.str())));

      lastmin_.push_back(0);
      lastmax_.push_back(1023); 

      gui->execute(id.c_str() + string(" make_min_max " + str4.str()));

      uis_.set(uis_.get() + 1);
  }
  else if( args[1] == "remove_axis" ) 
  {
    uis_.reset();
    int i = uis_.get()-1;
    ostringstream str;
    str << i;
    vector<GuiInt*>::iterator iter = mins_.end();
    vector<GuiInt*>::iterator iter2 = maxs_.end();
    vector<GuiInt*>::iterator iter3 = absmaxs_.end();
    vector<int>::iterator iter4 = lastmin_.end();
    vector<int>::iterator iter5 = lastmax_.end();
    mins_.erase(iter, iter);
    maxs_.erase(iter2, iter2);
    absmaxs_.erase(iter3, iter3);
    
    lastmin_.erase(iter4, iter4);
    lastmax_.erase(iter5, iter5);
    
    gui->execute(id.c_str() + string(" clear_axis " + str.str()));
    uis_.set(uis_.get() - 1);
  }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}
