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
 *  ConductivitySearch.cc:  Search for the optimal conductivity
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Institute
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Array2.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Math/Gaussian.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <BioPSE/Core/Algorithms/NumApproximation/BuildFEMatrix.h>
#include <iostream>
using std::endl;
#include <stdio.h>
#include <math.h>


namespace BioPSE {

using namespace SCIRun;

class ConductivitySearch : public Module {    
  FieldIPort     *mesh_iport_;
  MatrixIPort    *cond_params_iport_;
  MatrixIPort    *misfit_iport_;

  FieldOPort     *mesh_oport_;
  MatrixOPort    *cond_vector_oport_;
  MatrixOPort    *fem_mat_oport_;

  FieldHandle mesh_in_;
  FieldHandle mesh_out_;
  MatrixHandle cond_params_;
  MatrixHandle cond_vector_;
  MatrixHandle fem_mat_;

  MatrixHandle AmatH_;
  Array1<Array1<double> > data_basis_;
  Array1<double> misfit_;
  Array2<double> conductivities_;
  Array1<int> cell_visited_;
  Array1<double> cell_err_;
  Array1<Vector> cell_dir_;  
  int seed_counter_;
  string state_;
  int stop_search_;
  int last_intermediate_;
  Mutex mylock_;
  MusilRNG* mr;

  static int NDIM_;
  static int NSEEDS_;
  static int NCONDUCTIVITIES_;
  static int MAX_EVALS_;
  static double CONVERGENCE_;
  static double OUT_OF_BOUNDS_MISFIT_;

  void build_basis_matrices();
  void initialize_search();
  MatrixHandle build_composite_matrix(int which_conductivity);
  void send_and_get_data(int which_conductivity);
  int pre_search();
  void eval_test_conductivity();
  double simplex_step(Array1<double>& sum, double factor, int worst);
  void simplex_search();
  void read_mesh_and_cond_param_ports(int &valid_data, int &new_data);
public:
  GuiInt seed_gui_;
  ConductivitySearch(GuiContext *context);
  virtual ~ConductivitySearch();
  virtual void execute();
  virtual void tcl_command( GuiArgs&, void * );
};


DECLARE_MAKER(ConductivitySearch)


int ConductivitySearch::NDIM_ = 3;
int ConductivitySearch::NSEEDS_ = 4;
int ConductivitySearch::NCONDUCTIVITIES_ = 5;
int ConductivitySearch::MAX_EVALS_ = 100;
double ConductivitySearch::CONVERGENCE_ = 0.001;
double ConductivitySearch::OUT_OF_BOUNDS_MISFIT_ = 1000000;

ConductivitySearch::ConductivitySearch(GuiContext *context)
  : Module("ConductivitySearch",context, Filter, "Inverse", "BioPSE"), 
  mylock_("pause lock for ConductivitySearch"), 
  seed_gui_(context->subVar("seed_gui"))
{
  mylock_.unlock();
  state_ = "SEEDING";
  stop_search_ = 0;
  seed_counter_ = 0;
}

ConductivitySearch::~ConductivitySearch(){}


void ConductivitySearch::build_basis_matrices() {
  TetVolFieldIntHandle tviH;
  tviH = dynamic_cast<TetVolField<int> *>(mesh_in_.get_rep());
  TetVolFieldTensorHandle tvtH;
  Tensor zero(0);
  Tensor identity(1);

  double unitsScale = 1;
  string units;
  if (mesh_in_->get_property("units", units)) {
    if (units == "mm") unitsScale = 1./1000;
    else if (units == "cm") unitsScale = 1./100;
    else if (units == "dm") unitsScale = 1./10;
    else if (units == "m") unitsScale = 1./1;
    else
    {
      warning("Didn't recognize units of mesh: " + units + ".");
    }
  }

  MatrixHandle aH;
  vector<pair<string, Tensor> > tens(NDIM_, pair<string, Tensor>("", zero));
  BuildFEMatrix::build_FEMatrix(tviH, tvtH, true, tens, aH, unitsScale);
  AmatH_ = aH;
  AmatH_.detach(); // this will be our global "shape" information
  
  data_basis_.resize(NDIM_);
  for (int i=0; i<NDIM_; i++) {
    tens[i].first=to_string(i);
    tens[i].second=identity;
    BuildFEMatrix::build_FEMatrix(tviH, tvtH, true, tens, aH, unitsScale);
    SparseRowMatrix *m = dynamic_cast<SparseRowMatrix*>(aH.get_rep());
    data_basis_[i].resize(m->nnz);
    for (int j=0; j<m->nnz; j++)
      data_basis_[i][j] = m->a[j];
    tens[i].second=zero;
  }    
}


//! Initialization sets up our conductivity search matrix, our misfit vector,
//!   and our stiffness matrix frame

void ConductivitySearch::initialize_search() {
  // fill in our random conductivities and build our stiffness matrix frame
  misfit_.resize(NCONDUCTIVITIES_);
  conductivities_.resize(NCONDUCTIVITIES_, NDIM_);

  ColumnMatrix *cm;
  cond_vector_ = cm = new ColumnMatrix(NDIM_*2);
  int c;
  for (int c=0; c<NDIM_*2; c++) (*cm)[c]=0;
  vector<pair<string, Tensor> > conds;
  if (mesh_in_->get_property("conductivity_table", conds))
    for (c=NDIM_; c<NDIM_*2; c++) (*cm)[c]=conds[c-NDIM_].second.mat_[0][0];

  int seed = seed_gui_.get();
  seed_gui_.set(seed+1);
  mr = new MusilRNG(seed);
  (*mr)();  // first number isn't really random

  // load Gaussian distribution of conductivity values into our search matrix
  int i, j;
  for (i=0; i<NDIM_; i++) {
    double val, min, max, sigma, avg;
    avg=(*(cond_params_.get_rep()))[i][0];
    sigma=(*(cond_params_.get_rep()))[i][1];
    min=(*(cond_params_.get_rep()))[i][2];
    max=(*(cond_params_.get_rep()))[i][3];
    Gaussian g(avg, sigma);
    for (j=0; j<NDIM_+2; j++) {
      do {
	val= conductivities_(j,i) = g.rand();
      } while (val > max || val < min);
    }
  }
  build_basis_matrices();
}


//! Scale the basis matrix data by the conductivities and sum

MatrixHandle ConductivitySearch::build_composite_matrix(int 
							which_conductivity) {
  MatrixHandle fem_mat = AmatH_;
  fem_mat.detach();
  SparseRowMatrix *m = dynamic_cast<SparseRowMatrix*>(fem_mat.get_rep());
  double *sum = m->a;
  for (int i=0; i<NDIM_; i++) {
    double weight = conductivities_(which_conductivity,i);
    for (int j=0; j<data_basis_[i].size(); j++)
      sum[j] += weight*data_basis_[i][j];
  }
  return fem_mat;
}


//! Find the misfit and optimal orientation for a single conductivity

void ConductivitySearch::send_and_get_data(int which_conductivity) {
  if (!mylock_.tryLock()) {
    mylock_.lock();
    mylock_.unlock();
  } else {
    mylock_.unlock();
  }

  // build the new conductivity vector, the new mesh (just new tensors), 
  // and the new fem matrix
  cond_vector_.detach();
  mesh_out_ = mesh_in_;
  mesh_out_.detach();
  vector<pair<string, Tensor> > conds;
  ColumnMatrix *cm = dynamic_cast<ColumnMatrix*>(cond_vector_.get_rep());
  int i;
  for (i=0; i<NDIM_; i++) {
    double c=conductivities_(which_conductivity, i);
    conds.push_back(pair<string, Tensor>(to_string(i), Tensor(c)));
    (*cm)[i]=c;
  }
  mesh_out_->set_property("conductivity_table", conds, true);

  fem_mat_ = build_composite_matrix(which_conductivity);
  
  // send out data
  mesh_oport_->send_intermediate(mesh_out_);
  cond_vector_oport_->send_intermediate(cond_vector_);
  fem_mat_oport_->send_intermediate(fem_mat_);
  last_intermediate_=1;

  // read back data, and set the caches and search matrix
  MatrixHandle mH;
  Matrix* m;
  if (!misfit_iport_->get(mH) || !(m = mH.get_rep())) {
    error("ConductivitySearch::failed to read back error");
    return;
  }
  misfit_[which_conductivity]=(*m)[0][0];
}  


//! pre_search gets called once for each seed conductivity.  It sends out
//!   one of the seeds, reads back the results, and fills the data
//!   into the caches and the search matrix
//! return "fail" if any seeds are out of range or if we don't get
//!   back a misfit after a send 

int ConductivitySearch::pre_search() {
  if (seed_counter_ == 0) {
    initialize_search();
  }

  send_and_get_data(seed_counter_);
  seed_counter_++;

  // done seeding, prepare for search phase
  if (seed_counter_ == NSEEDS_) {
    seed_counter_ = 0;
    state_ = "START_SEARCHING";
  }
  return 1;
}


//! Evaluate a test conductivity.

void ConductivitySearch::eval_test_conductivity() {
  
  int in_range=1;
  for (int i=0; i<NDIM_; i++)
    if (conductivities_(NSEEDS_,i)<(*(cond_params_.get_rep()))[i][2]) 
      in_range=0;
    else if (conductivities_(NSEEDS_,i)>(*(cond_params_.get_rep()))[i][3])
      in_range=0;

  if (in_range) {
    send_and_get_data(NSEEDS_);
  } else {
    misfit_[NSEEDS_]=OUT_OF_BOUNDS_MISFIT_;
  }
}


//! Take a single simplex step.  Evaluate a new position -- if it's
//! better then an existing vertex, swap them.

double ConductivitySearch::simplex_step(Array1<double>& sum, double factor,
					int worst) {
  double factor1 = (1 - factor)/NDIM_;
  double factor2 = factor1-factor;
  int i;
  for (i=0; i<NDIM_; i++) 
    conductivities_(NSEEDS_,i) = sum[i]*factor1 - 
      conductivities_(worst,i)*factor2;

  // evaluate the new guess
  eval_test_conductivity();

  // if this is better, swap it with the worst one
  if (misfit_[NSEEDS_] < misfit_[worst]) {
    misfit_[worst] = misfit_[NSEEDS_];
    for (i=0; i<NDIM_; i++) {
      sum[i] = sum[i] + conductivities_(NSEEDS_,i)-conductivities_(worst,i);
      conductivities_(worst,i) = conductivities_(NSEEDS_,i);
    }
  }
  return misfit_[NSEEDS_];
}


//! The simplex has been constructed -- now let's search for a minimal misfit

void ConductivitySearch::simplex_search() {
  Array1<double> sum(NDIM_); // sum of the entries in the search matrix rows
  sum.initialize(0);
  int i, j;
  for (i=0; i<NSEEDS_; i++) 
    for (j=0; j<NDIM_; j++)
      sum[j]+=conductivities_(i,j); 

  double relative_tolerance;
  int num_evals = 0;

  while(1) {
    int best, worst, next_worst;
    best = 0;
    if (misfit_[0] > misfit_[1]) {
      worst = 0;
      next_worst = 1;
    } else {
      worst = 1;
      next_worst = 0;
    }
    for (i=0; i<NSEEDS_; i++) {
      if (misfit_[i] <= misfit_[best]) best=i;
      if (misfit_[i] > misfit_[worst]) {
	next_worst = worst;
	worst = i;
      } else 
	if (misfit_[i] > misfit_[next_worst] && (i != worst)) 
	  next_worst=i;
      relative_tolerance = 2*(misfit_[worst]-misfit_[best])/
	(misfit_[worst]+misfit_[best]);
    }

    if ((relative_tolerance < CONVERGENCE_) || 
	(num_evals > MAX_EVALS_) || (stop_search_)) 
      break;

    double step_misfit = simplex_step(sum, -1, worst);
    num_evals++;
    if (step_misfit <= misfit_[best]) {
      step_misfit = simplex_step(sum, 2, worst);
      num_evals++;
    } else if (step_misfit >= misfit_[worst]) {
      double old_misfit = misfit_[worst];
      step_misfit = simplex_step(sum, 0.5, worst);
      num_evals++;
      if (step_misfit >= old_misfit) {
	for (i=0; i<NSEEDS_; i++) {
	  if (i != best) {
	    for (j=0; j<NDIM_; j++)
	      conductivities_(i,j) = conductivities_(NSEEDS_,j) = 
		0.5 * (conductivities_(i,j) + conductivities_(best,j));
	    eval_test_conductivity();
	    misfit_[i] = misfit_[NSEEDS_];
	    num_evals++;
	  }
	}
      }
      sum.initialize(0);
      for (i=0; i<NSEEDS_; i++) {
	for (j=0; j<NDIM_; j++)
	  sum[j]+=conductivities_(i,j); 
      }
    }
    if ((num_evals % 10) == 0) {
      msgStream_ << "ConductivitySearch -- Iter "<<num_evals<<":\n";
      for (i=0; i<NSEEDS_; i++) {
	msgStream_ << "\t";
	for (j=0; j<NDIM_; j++) {
	  msgStream_ << conductivities_(i,j) << " ";
	}
	msgStream_ << "\n";
      }
    }
  }
  ColumnMatrix *cm = dynamic_cast<ColumnMatrix*>(cond_vector_.get_rep());
  msgStream_ << "ConductivitySearch -- Original conductivities: \n\t";
  for (i=NDIM_; i<NDIM_*2; i++) msgStream_ << (*cm)[i] << " ";
  msgStream_ << "\nConductivitiySearch -- Final conductivities: \n";
  for (i=0; i<NSEEDS_; i++) {
    msgStream_ << "\t";
    for (j=0; j<NDIM_; j++) {
      msgStream_ << conductivities_(i,j) << " ";
    }
    msgStream_ << "(error="<<misfit_[i]<<")\n";
  }
}


//! Read the input fields.  Check whether the inputs are valid,
//! and whether they've changed since last time.

void ConductivitySearch::read_mesh_and_cond_param_ports(int &valid_data, 
							int &new_data) {
  FieldHandle mesh;
  valid_data=1;
  new_data=0;
  if (mesh_iport_->get(mesh) && mesh.get_rep() &&
      (mesh->get_type_name(0) == "TetVolField") &&
      (mesh->get_type_name(1) == "int")) {
    if (!mesh_in_.get_rep() || (mesh_in_->generation != mesh->generation)) {
      new_data=1;
      mesh_in_=mesh;
    } else {
      remark("Same VolumeMesh as previous run.");
    }
  } else {
    valid_data=0;
    warning("Didn't get a valid VolumeMesh.");
  }

  TetVolField<int> *meshTV = dynamic_cast<TetVolField<int> *>(mesh.get_rep());
  Array1<Tensor> tens;
  pair<int,int> minmax;
  minmax.second=1;
  if (!field_minmax(*meshTV, minmax)) valid_data=0;
  NDIM_ = minmax.second+1;
  NSEEDS_ = minmax.second+2;
  NCONDUCTIVITIES_ = minmax.second+3;

  MatrixHandle cond_params;
  if (cond_params_iport_->get(cond_params) && cond_params.get_rep() &&
      (cond_params->nrows() == NDIM_) && (cond_params->ncols() == 4)) {
    if (!cond_params_.get_rep() || 
	(cond_params_->generation != cond_params->generation)) {
      new_data = 1;
      cond_params_=cond_params;
    } else {
      remark("Same ConductivityParams as before.");
    }
  } else {
    valid_data=0;
    warning("Didn't get valid ConductivityParams.");
  }
}


//! If we have an old solution and the inputs haven't changed, send that
//! one.  Otherwise, if the input is valid, run a simplex search to find
//! a conductivity with minimal misfit.

void ConductivitySearch::execute() {
  int valid_data, new_data;
  mesh_iport_ = (FieldIPort *)get_iport("FiniteElementMesh");
  cond_params_iport_ = (MatrixIPort *)get_iport("ConductivityParameters");
  misfit_iport_ = (MatrixIPort *)get_iport("TestMisfit");

  mesh_oport_ = (FieldOPort *)get_oport("FiniteElementMesh");
  cond_vector_oport_ = (MatrixOPort *)get_oport("OldAndNewConductivities");
  fem_mat_oport_ = (MatrixOPort *)get_oport("FiniteElementMatrix");

  if (!mesh_iport_) {
    error("Unable to initialize iport 'FiniteElementMesh'.");
    return;
  }
  if (!cond_params_iport_) {
    error("Unable to initialize iport 'ConductivityParameters'.");
    return;
  }
  if (!misfit_iport_) {
    error("Unable to initialize iport 'TestMisfit'.");
    return;
  }
  if (!mesh_oport_) {
    error("Unable to initialize oport 'FiniteElementMesh'.");
    return;
  }
  if (!cond_vector_oport_) {
    error("Unable to initialize oport 'OldAndNewConductivities'.");
    return;
  }
  if (!fem_mat_oport_) {
    error("Unable to initialize oport 'FiniteElementMatrix'.");
    return;
  }
  
  read_mesh_and_cond_param_ports(valid_data, new_data);
  if (!valid_data) return;
  if (!new_data) {
    if (mesh_out_.get_rep()) { // if we have valid old data
      // send old data and clear ports
      remark("Sending old data.");
      mesh_oport_->send(mesh_out_);
      cond_vector_oport_->send(cond_vector_);
      fem_mat_oport_->send(fem_mat_);
      MatrixHandle dummy_mat;
      misfit_iport_->get(dummy_mat);
      return;
    } else {
      return;
    }
  }

  last_intermediate_=0;

  // we have new, valid data -- run the simplex search
  while (1) {
    if (state_ == "SEEDING") {
      if (!pre_search()) break;
    } else if (state_ == "START_SEARCHING") {
      simplex_search();
      state_ = "DONE";
    }
    if (stop_search_ || state_ == "DONE") break;
  }
  if (last_intermediate_) { // last sends were send_intermediates
    // gotta do final sends and clear the ports
    mesh_oport_->send(mesh_out_);
    cond_vector_oport_->send(cond_vector_);
    fem_mat_oport_->send(fem_mat_);
    MatrixHandle dummy_mat;
    misfit_iport_->get(dummy_mat);
  }
  state_ = "SEEDING";
  stop_search_=0;
  seed_counter_=0;
}


//! Commands invoked from the Gui.  Pause/unpause/stop the search.

void
ConductivitySearch::tcl_command(GuiArgs& args, void* userdata)
{
  if (args[1] == "pause") {
    if (mylock_.tryLock())
      msgStream_ << "Pausing..."<<endl;
    else 
      msgStream_ << "Can't lock -- already locked"<<endl;
  } else if (args[1] == "unpause") {
    if (mylock_.tryLock())
      msgStream_ << "Can't unlock -- already unlocked"<<endl;
    else
      msgStream_ << "Unpausing"<<endl;
    mylock_.unlock();
  } else if (args[1] == "stop") {
    stop_search_=1;
  } else if (args[1] == "exec") {
    mesh_in_=0;
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace BioPSE
