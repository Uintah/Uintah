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
 *  DipoleSearch.cc:  Search for the optimal dipole
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
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <iostream>
using std::endl;
#include <stdio.h>
#include <math.h>


namespace BioPSE {
using namespace SCIRun;

class DipoleSearch : public Module {    
  FieldIPort     *seeds_iport_;
  FieldIPort     *mesh_iport_;
  MatrixIPort    *misfit_iport_;
  MatrixIPort    *dir_iport_;

  MatrixOPort    *leadfield_select_oport_;
  FieldOPort     *simplex_oport_;
  FieldOPort     *dipole_oport_;

  FieldHandle seedsH_;
  FieldHandle meshH_;
  TetVolMeshHandle vol_mesh_;

  MatrixHandle leadfield_selectH_;
  FieldHandle simplexH_;
  FieldHandle dipoleH_;
  
  int seed_counter_;
  string state_;
  Array1<double> misfit_;
  Array2<double> dipoles_;
  Array1<int> cell_visited_;
  Array1<double> cell_err_;
  Array1<Vector> cell_dir_;  
  int use_cache_;
  int stop_search_;
  int last_intermediate_;
  Mutex mylock_;

  static int NDIM_;
  static int NSEEDS_;
  static int NDIPOLES_;
  static int MAX_EVALS_;
  static double CONVERGENCE_;
  static double OUT_OF_BOUNDS_MISFIT_;

  void initialize_search();
  void send_and_get_data(int which_dipole, TetVolMesh::Cell::index_type ci);
  int pre_search();
  Vector eval_test_dipole();
  double simplex_step(Array1<double>& sum, double factor, int worst);
  void simplex_search();
  void read_field_ports(int &valid_data, int &new_data);
  void organize_last_send();
  int find_better_neighbor(int best, Array1<double>& sum);
public:
  GuiInt use_cache_gui_;
  DipoleSearch(GuiContext *context);
  virtual ~DipoleSearch();
  virtual void execute();
  virtual void tcl_command( GuiArgs&, void * );
};


DECLARE_MAKER(DipoleSearch)



int DipoleSearch::NDIM_ = 3;
int DipoleSearch::NSEEDS_ = 4;
int DipoleSearch::NDIPOLES_ = 5;
int DipoleSearch::MAX_EVALS_ = 1001;
double DipoleSearch::CONVERGENCE_ = 0.0001;
double DipoleSearch::OUT_OF_BOUNDS_MISFIT_ = 1000000;

DipoleSearch::DipoleSearch(GuiContext *context)
  : Module("DipoleSearch", context, Filter, "Inverse", "BioPSE"), 
  mylock_("pause lock for DipoleSearch"), 
  use_cache_gui_(context->subVar("use_cache_gui_"))
{
  mylock_.unlock();
  state_ = "SEEDING";
  stop_search_ = 0;
  seed_counter_ = 0;
}

DipoleSearch::~DipoleSearch(){}


//! Initialization sets up our dipole search matrix, and misfit vector, and 
//!   resizes and initializes our caching vectors

void DipoleSearch::initialize_search() {
  // cast the mesh based class up to a tetvolmesh
  PointCloudMesh *seeds_mesh =
    (PointCloudMesh*)dynamic_cast<PointCloudMesh*>(seedsH_->mesh().get_rep());

  // iterate through the nodes and copy the positions into our 
  //  simplex search matrix (`dipoles')
  PointCloudMesh::Node::iterator ni; seeds_mesh->begin(ni);
  PointCloudMesh::Node::iterator nie; seeds_mesh->end(nie);
  misfit_.resize(NDIPOLES_);
  dipoles_.resize(NDIPOLES_, NDIM_+3);
  for (int nc=0; ni != nie; ++ni, nc++) {
    Point p;
    seeds_mesh->get_center(p, *ni);
    dipoles_(nc,0)=p.x(); dipoles_(nc,1)=p.y(); dipoles_(nc,2)=p.z();
    dipoles_(nc,3)=0; dipoles_(nc,4)=0; dipoles_(nc,5)=1;
  }
  // this last dipole entry contains our test dipole
  int j;
  for (j=0; j<NDIM_+3; j++) 
    dipoles_(NSEEDS_,j)=0;

  TetVolMesh::Cell::size_type csize;
  vol_mesh_->size(csize);
  cell_visited_.resize(csize);
  cell_err_.resize(csize);
  cell_dir_.resize(csize);
  cell_visited_.initialize(0);
  use_cache_=use_cache_gui_.get();
}


//! Find the misfit and optimal orientation for a single dipole

void DipoleSearch::send_and_get_data(int which_dipole, 
				     TetVolMesh::Cell::index_type ci) {
  if (!mylock_.tryLock()) {
    msgStream_ << "Thread is paused\n";
    mylock_.lock();
    mylock_.unlock();
    msgStream_ << "Thread is unpaused\n";
  } else {
    mylock_.unlock();
  }

  int j;

  // build columns for MatrixSelectVec
  // each column is an interpolant vector or index/weight pairs
  // we just have one entry for each -- index=leadFieldColumn, weight=1
  DenseMatrix *leadfield_select_out = scinew DenseMatrix(2, 3);
  for (j=0; j<3; j++) {
    (*leadfield_select_out)[0][j] = ci*3+j;
    (*leadfield_select_out)[1][j] = 1;
  }
  
  PointCloudMeshHandle pcm = scinew PointCloudMesh;
  for (j=0; j<NSEEDS_; j++)
    pcm->add_point(Point(dipoles_(j,0), dipoles_(j,1), dipoles_(j,2)));
  PointCloudField<Vector> *pcv = scinew PointCloudField<Vector>(pcm, Field::NODE);
  for (j=0; j<NSEEDS_; j++)
    pcv->fdata()[j] = Vector(dipoles_(j,3), dipoles_(j,4), dipoles_(j,5));

  pcm = scinew PointCloudMesh;
  pcm->add_point(Point(dipoles_(which_dipole, 0), dipoles_(which_dipole, 1),
		       dipoles_(which_dipole, 2)));
  PointCloudField<Vector> *pcd = scinew PointCloudField<Vector>(pcm, Field::NODE);
  
  // send out data
  leadfield_selectH_ = leadfield_select_out;
  leadfield_select_oport_->send_intermediate(leadfield_selectH_);
  simplexH_ = pcv;
  simplex_oport_->send_intermediate(simplexH_);
  dipoleH_ = pcd;
  dipole_oport_->send_intermediate(dipoleH_);
  last_intermediate_=1;

  // read back data, and set the caches and search matrix
  MatrixHandle mH;
  Matrix* m;
  if (!misfit_iport_->get(mH) || !(m = mH.get_rep())) {
    error("DipoleSearch::failed to read back error");
    return;
  }
  cell_err_[ci]=misfit_[which_dipole]=(*m)[0][0];
  if (!dir_iport_->get(mH) || !(m=mH.get_rep()) || mH->nrows()<3) {
    error("DipoleSearch::failed to read back orientation");
    return;
  }
  cell_dir_[ci]=Vector((*m)[0][0], (*m)[1][0], (*m)[2][0]);
  for (j=0; j<3; j++)
    dipoles_(which_dipole,j+3)=(*m)[j][0];
}  


//! pre_search gets called once for each seed dipole.  It sends out
//!   one of the seeds, reads back the results, and fills the data
//!   into the caches and the search matrix
//! return "fail" if any seeds are out of mesh or if we don't get
//!   back a misfit or optimal orientation after a send 

int DipoleSearch::pre_search() {
  if (seed_counter_ == 0) {
    initialize_search();
  }

  // send out a seed dipole and get back the misfit and the optimal orientation
  TetVolMesh::Cell::index_type ci;
  if (!vol_mesh_->locate(ci, Point(dipoles_(seed_counter_,0), 
				   dipoles_(seed_counter_,1), 
				   dipoles_(seed_counter_,2)))) {
    warning("Seedpoint " + to_string(seed_counter_) + " is outside of mesh.");
    return 0;
  } 
  if (cell_visited_[ci]) {
    warning("Redundant seedpoints found.");
    misfit_[seed_counter_]=cell_err_[ci];
  } else {
    cell_visited_[ci]=1;
    send_and_get_data(seed_counter_, ci);
  }
  seed_counter_++;

  // done seeding, prepare for search phase
  if (seed_counter_ == NSEEDS_) {
    seed_counter_ = 0;
    state_ = "START_SEARCHING";
  }
  return 1;
}


//! Evaluate a test dipole.  Return the optimal orientation.

Vector DipoleSearch::eval_test_dipole() {
  TetVolMesh::Cell::index_type ci;
  if (vol_mesh_->locate(ci, Point(dipoles_(NSEEDS_,0), 
				  dipoles_(NSEEDS_,1), 
				  dipoles_(NSEEDS_,2)))) {
    if (!cell_visited_[ci] || !use_cache_) {
      cell_visited_[ci]=1;
      send_and_get_data(NSEEDS_, ci);
    } else {
      misfit_[NSEEDS_]=cell_err_[ci];
    }
    return (cell_dir_[ci]);
  } else {
    misfit_[NSEEDS_]=OUT_OF_BOUNDS_MISFIT_;
    return (Vector(0,0,1));
  }
}


//! Check to see if any of the neighbors of the "best" dipole are better.

int DipoleSearch::find_better_neighbor(int best, Array1<double>& sum) {
  Point p(dipoles_(best, 0), dipoles_(best, 1), dipoles_(best, 2));
  TetVolMesh::Cell::index_type ci;
  vol_mesh_->locate(ci, p);
  TetVolMesh::Cell::array_type ca;
  vol_mesh_->synchronize(Mesh::FACE_NEIGHBORS_E);
  vol_mesh_->get_neighbors(ca, ci);
  for (unsigned int n=0; n<ca.size(); n++) {
    Point p1;
    vol_mesh_->get_center(p1, ca[n]);
    dipoles_(NSEEDS_,0)=p1.x();
    dipoles_(NSEEDS_,1)=p1.y();
    dipoles_(NSEEDS_,2)=p1.z();
    eval_test_dipole();
    if (misfit_[NSEEDS_] < misfit_[best]) {
      misfit_[best] = misfit_[NSEEDS_];
      int i;
      for (i=0; i<NDIM_; i++) {
	sum[i] = sum[i] + dipoles_(NSEEDS_,i)-dipoles_(best,i);
	dipoles_(best,i) = dipoles_(NSEEDS_,i);
      }
      for (i=NDIM_; i<NDIM_+3; i++)
	dipoles_(best,i) = dipoles_(NSEEDS_,i);  // copy orientation data
    }
    return 1;
  }
  return 0;
}


//! Take a single simplex step.  Evaluate a new position -- if it's
//! better then an existing vertex, swap them.

double DipoleSearch::simplex_step(Array1<double>& sum, double factor,
				  int worst) {
  double factor1 = (1 - factor)/NDIM_;
  double factor2 = factor1-factor;
  int i;
  for (i=0; i<NDIM_; i++) 
    dipoles_(NSEEDS_,i) = sum[i]*factor1 - dipoles_(worst,i)*factor2;

  // evaluate the new guess
  eval_test_dipole();

  // if this is better, swap it with the worst one
  if (misfit_[NSEEDS_] < misfit_[worst]) {
    misfit_[worst] = misfit_[NSEEDS_];
    for (i=0; i<NDIM_; i++) {
      sum[i] = sum[i] + dipoles_(NSEEDS_,i)-dipoles_(worst,i);
      dipoles_(worst,i) = dipoles_(NSEEDS_,i);
    }
    for (i=NDIM_; i<NDIM_+3; i++)
      dipoles_(worst,i) = dipoles_(NSEEDS_,i);  // copy orientation data
  }

  return misfit_[NSEEDS_];
}

//! The simplex has been constructed -- now let's search for a minimal misfit

void DipoleSearch::simplex_search() {
  Array1<double> sum(NDIM_);  // sum of the entries in the search matrix rows
  sum.initialize(0);
  int i, j;
  for (i=0; i<NSEEDS_; i++) 
    for (j=0; j<NDIM_; j++)
      sum[j]+=dipoles_(i,j); 

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
    int i;
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

    if (num_evals > MAX_EVALS_ || stop_search_) break;

    // make sure all of our neighbors are worse than us...
    if (relative_tolerance < CONVERGENCE_) {
      if (misfit_[best]>1.e-12 && find_better_neighbor(best, sum)) {
	num_evals++;
	continue;
      }
      break;
    }

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
	    int j;
	    for (j=0; j<NDIM_; j++)
	      dipoles_(i,j) = dipoles_(NSEEDS_,j) = 
		0.5 * (dipoles_(i,j) + dipoles_(best,j));
	    Vector dir = eval_test_dipole();
	    misfit_[i] = misfit_[NSEEDS_];
	    dipoles_(i,3) = dir.x(); 
	    dipoles_(i,4) = dir.y(); 
	    dipoles_(i,5) = dir.z(); 
	    num_evals++;
	  }
	}
      }
      sum.initialize(0);
      for (i=0; i<NSEEDS_; i++) {
	for (j=0; j<NDIM_; j++)
	  sum[j]+=dipoles_(i,j); 
      }
    }
  }
  msgStream_ << "DipoleSearch -- num_evals = "<<num_evals << "\n";
}


//! Read the input fields.  Check whether the inputs are valid,
//! and whether they've changed since last time.

void DipoleSearch::read_field_ports(int &valid_data, int &new_data) {
  FieldHandle mesh;
  valid_data=1;
  new_data=0;
  if (mesh_iport_->get(mesh) && mesh.get_rep() &&
      (mesh->get_type_name(0) == "TetVolField")) {
    if (!meshH_.get_rep() || (meshH_->generation != mesh->generation)) {
      new_data=1;
      meshH_=mesh;
      // cast the mesh base class up to a tetvolmesh
      vol_mesh_=
	(TetVolMesh*)dynamic_cast<TetVolMesh*>(mesh->mesh().get_rep());
    } else {
      remark("Same VolumeMesh as previous run.");
    }
  } else {
    valid_data=0;
    remark("Didn't get a valid VolumeMesh.");
  }
  
  FieldHandle seeds;    
  if (!seeds_iport_->get(seeds)) {
    warning("No input seeds.");
    valid_data=0;
  } else if (!seeds.get_rep()) {
    warning("Empty seeds handle.");
    valid_data=0;
  } else if (seeds->get_type_name(-1) != "PointCloudField<double> ") {
    warning("Seeds typename should have been PointCloudField<double>.");
    valid_data=0;
  } else {
    PointCloudField<double> *d=dynamic_cast<PointCloudField<double> *>(seeds.get_rep());
    PointCloudMesh::Node::size_type nsize; d->get_typed_mesh()->size(nsize);
    if (nsize != (unsigned int)NSEEDS_){
      msgStream_ << "Got "<< nsize <<" seeds, instead of "<<NSEEDS_<<"\n";
      valid_data=0;
    } else if (!seedsH_.get_rep() || 
	       (seedsH_->generation != seeds->generation)) {
      new_data = 1;
      seedsH_=seeds;
    } else {
      remark("Using same seeds as before.");
    }
  }
}

void DipoleSearch::organize_last_send() {
  int bestIdx=0;
  double bestMisfit = misfit_[0];
  int i;
  for (i=1; i<misfit_.size(); i++)
    if (misfit_[i] < bestMisfit) {
      bestMisfit = misfit_[i];
      bestIdx = i;
    }

  Point best_pt(dipoles_(bestIdx,0), dipoles_(bestIdx,1), dipoles_(bestIdx,2));
  TetVolMesh::Cell::index_type best_cell_idx;
  vol_mesh_->locate(best_cell_idx, best_pt);

  msgStream_ << "DipoleSearch -- the dipole was found in cell " << best_cell_idx << "\n    at position " << best_pt << " with a misfit of " << bestMisfit << "\n";

  DenseMatrix *leadfield_select_out = scinew DenseMatrix(2, 3);
  for (i=0; i<3; i++) {
    (*leadfield_select_out)[0][i] = best_cell_idx*3+i;
    (*leadfield_select_out)[1][i] = 1;
  }
  leadfield_selectH_ = leadfield_select_out;
  PointCloudMeshHandle pcm = scinew PointCloudMesh;
  pcm->add_point(best_pt);
  PointCloudField<Vector> *pcd = scinew PointCloudField<Vector>(pcm, Field::NODE);
  pcd->fdata()[0]=Vector(dipoles_(bestIdx,3), dipoles_(bestIdx,4), dipoles_(bestIdx,5));
  dipoleH_ = pcd;
}

//! If we have an old solution and the inputs haven't changed, send that
//! one.  Otherwise, if the input is valid, run a simplex search to find
//! a dipole with minimal misfit.

void DipoleSearch::execute() {
  int valid_data, new_data;
  // point cloud of vectors -- the seed positions/orientations for our search
  seeds_iport_ = (FieldIPort *)get_iport("DipoleSeeds");
  // domain of search -- used for constraining and for caching misfits
  mesh_iport_ = (FieldIPort *)get_iport("TetMesh");
  // the computed misfit for the latest test dipole
  misfit_iport_ = (MatrixIPort *)get_iport("TestMisfit");
  // optimal orientation for the test dipole
  dir_iport_ = (MatrixIPort *)get_iport("TestDirection");

  // matrix of selection columns (for MatrixSelectVector) to pull out the
  //    appropriate x/y/z columns from the leadfield matrix
  leadfield_select_oport_ = 
    (MatrixOPort *)get_oport("LeadFieldSelectionMatrix");
  // point cloud of vectors -- the latest simplex (for vis)
  simplex_oport_ = (FieldOPort *)get_oport("DipoleSimplex");
  // point cloud of one vector, just the test dipole (for vis)
  dipole_oport_ = (FieldOPort *)get_oport("TestDipole");

  if (!seeds_iport_) {
    error("Unable to initialize iport 'DipoleSeeds'.");
    return;
  }
  if (!mesh_iport_) {
    error("Unable to initialize iport 'TetMesh'.");
    return;
  }
  if (!misfit_iport_) {
    error("Unable to initialize iport 'TestMisfit'.");
    return;
  }
  if (!dir_iport_) {
    error("Unable to initialize iport 'TestDirection'.");
    return;
  }
  if (!leadfield_select_oport_) {
    error("Unable to initialize oport 'LeadFieldSelectionMatrix'.");
    return;
  }
  if (!simplex_oport_) {
    error("Unable to initialize oport 'DipoleSimplex'.");
    return;
  }
  if (!dipole_oport_) {
    error("Unable to initialize oport 'TestDipole'.");
    return;
  }

  read_field_ports(valid_data, new_data);
  if (!valid_data) return;
  if (!new_data) {
    if (simplexH_.get_rep()) { // if we have valid old data
      // send old data and clear ports
      leadfield_select_oport_->send(leadfield_selectH_);
      simplex_oport_->send(simplexH_);
      dipole_oport_->send(dipoleH_);
      MatrixHandle dummy_mat;
      misfit_iport_->get(dummy_mat);
      dir_iport_->get(dummy_mat);
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
    organize_last_send();
    leadfield_select_oport_->send(leadfield_selectH_);
    simplex_oport_->send(simplexH_);
    dipole_oport_->send(dipoleH_);

    MatrixHandle dummy_mat;
    misfit_iport_->get(dummy_mat);
    dir_iport_->get(dummy_mat);
  }
  state_ = "SEEDING";
  stop_search_=0;
  seed_counter_=0;
}


//! Commands invoked from the Gui.  Pause/unpause/stop the search.

void DipoleSearch::tcl_command(GuiArgs& args, void* userdata)
{
  if (args[1] == "pause") {
    if (mylock_.tryLock())
      msgStream_ << "TCL initiating pause..."<<endl;
    else 
      msgStream_ << "Can't lock -- already locked"<<endl;
  } else if (args[1] == "unpause") {
    if (mylock_.tryLock())
      msgStream_ << "Can't unlock -- already unlocked"<<endl;
    else
      msgStream_ << "TCL initiating unpause..."<<endl;
    mylock_.unlock();
  } else if (args[1] == "stop") {
    stop_search_=1;
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace BioPSE
