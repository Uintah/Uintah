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
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <iostream>
using std::cerr;
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

  MatrixOPort    *x_oport_;
  MatrixOPort    *y_oport_;
  MatrixOPort    *z_oport_;
  FieldOPort     *simplex_oport_;
  FieldOPort     *dipole_oport_;

  FieldHandle seedsH_;
  FieldHandle meshH_;
  TetVolMeshHandle vol_mesh_;
  FieldHandle simplexH_;

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
public:
  GuiInt use_cache_gui_;
  DipoleSearch(const string& id);
  virtual ~DipoleSearch();
  virtual void execute();
  void tcl_command( TCLArgs&, void * );
};

extern "C" Module* make_DipoleSearch(const string& id) {
  return new DipoleSearch(id);
}

int DipoleSearch::NDIM_ = 3;
int DipoleSearch::NSEEDS_ = 4;
int DipoleSearch::NDIPOLES_ = 5;
int DipoleSearch::MAX_EVALS_ = 100;
double DipoleSearch::CONVERGENCE_ = 0.001;
double DipoleSearch::OUT_OF_BOUNDS_MISFIT_ = 1000000;

DipoleSearch::DipoleSearch(const string& id)
  : Module("DipoleSearch", id, Filter), 
  mylock_("pause lock for DipoleSearch"), 
  use_cache_gui_("use_cache_gui_",id,this)
{
  // point cloud of vectors -- the seed positions/orientations for our search
  seeds_iport_ = new FieldIPort(this, "DipoleSeeds",
				      FieldIPort::Atomic);
  add_iport(seeds_iport_);
  
  // domain of search -- used for constraining and for caching misfits
  mesh_iport_ = new FieldIPort(this,"TetMesh",
			      FieldIPort::Atomic);
  add_iport(mesh_iport_);

  // the computed misfit for the latest test dipole
  misfit_iport_ = new MatrixIPort(this, "TestMisfit",
				 MatrixIPort::Atomic);
  add_iport(misfit_iport_);
  
  // optimal orientation for the test dipole
  dir_iport_ = new MatrixIPort(this,"TestDirection",
			      MatrixIPort::Atomic);
  add_iport(dir_iport_);
  
  // column matrix of the position and x-orientation of the test dipole
  x_oport_ = new MatrixOPort(this, "TestDipoleX",
				MatrixIPort::Atomic);
  add_oport(x_oport_);
  
  // column matrix of the position and y-orientation of the test dipole
  y_oport_ = new MatrixOPort(this, "TestDipoleY",
				MatrixIPort::Atomic);
  add_oport(y_oport_);
  
  // column matrix of the position and z-orientation of the test dipole
  z_oport_ = new MatrixOPort(this, "TestDipoleZ",
				MatrixIPort::Atomic);
  add_oport(z_oport_);
  
  // point cloud of vectors -- the latest simplex (for vis)
  simplex_oport_ = new FieldOPort(this, "DipoleSimplex",
				FieldIPort::Atomic);
  add_oport(simplex_oport_);

  // point cloud of one vector, just the test dipole (for vis)
  dipole_oport_ = new FieldOPort(this, "TestDipole",
				FieldIPort::Atomic);
  add_oport(dipole_oport_);

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
  PointCloudMesh::Node::iterator ni = seeds_mesh->node_begin();
  misfit_.resize(NDIPOLES_);
  dipoles_.newsize(NDIPOLES_, NDIM_+3);
  for (int nc=0; ni != seeds_mesh->node_end(); ++ni, nc++) {
    Point p;
    seeds_mesh->get_center(p, *ni);
    dipoles_(nc,0)=p.x(); dipoles_(nc,1)=p.y(); dipoles_(nc,2)=p.z();
    dipoles_(nc,3)=0; dipoles_(nc,4)=0; dipoles_(nc,5)=1;
  }
  // this last dipole entry contains our test dipole
  int j;
  for (j=0; j<NDIM_+3; j++) 
    dipoles_(NSEEDS_,j)=0;

  cell_visited_.resize(vol_mesh_->cells_size());
  cell_err_.resize(vol_mesh_->cells_size());
  cell_dir_.resize(vol_mesh_->cells_size());
  cell_visited_.initialize(0);
  use_cache_=use_cache_gui_.get();
}


//! Find the misfit and optimal orientation for a single dipole

void DipoleSearch::send_and_get_data(int which_dipole, 
				     TetVolMesh::Cell::index_type ci) {
  if (!mylock_.tryLock()) {
    mylock_.lock();
    mylock_.unlock();
  } else {
    mylock_.unlock();
  }

  ColumnMatrix *x_out = scinew ColumnMatrix(7);
  ColumnMatrix *y_out = scinew ColumnMatrix(7);
  ColumnMatrix *z_out = scinew ColumnMatrix(7);
  int j;
  for (j=0; j<3; j++)
    (*x_out)[j] = (*y_out)[j] = (*z_out)[j] = dipoles_(which_dipole,j);
  for (j=3; j<6; j++)
    (*x_out)[j] = (*y_out)[j] = (*z_out)[j] = 0;
  (*x_out)[3] = (*y_out)[4] = (*z_out)[5] = 1;
  (*x_out)[6] = (*y_out)[6] = (*z_out)[6] = (double)ci;
  
  PointCloudMeshHandle pcm = scinew PointCloudMesh;
  for (j=0; j<NSEEDS_; j++)
    pcm->add_point(Point(dipoles_(j,0), dipoles_(j,1), dipoles_(j,2)));
  PointCloud<Vector> *pcv = scinew PointCloud<Vector>(pcm, Field::NODE);
  for (j=0; j<NSEEDS_; j++)
    pcv->fdata()[j] = Vector(dipoles_(j,3), dipoles_(j,4), dipoles_(j,5));

  pcm = scinew PointCloudMesh;
  pcm->add_point(Point(dipoles_(which_dipole, 0), dipoles_(which_dipole, 1),
		       dipoles_(which_dipole, 2)));
  PointCloud<double> *pcd = scinew PointCloud<double>(pcm, Field::NODE);
  
  // send out data
  x_oport_->send_intermediate(x_out);
  y_oport_->send_intermediate(y_out);
  z_oport_->send_intermediate(z_out);
  simplexH_ = pcv;
  simplex_oport_->send_intermediate(simplexH_);
  dipole_oport_->send_intermediate(pcd);
  last_intermediate_=1;

  // read back data, and set the caches and search matrix
  MatrixHandle mH;
  Matrix* m;
  if (!misfit_iport_->get(mH) || !(m = mH.get_rep())) {
    error("DipoleSearch::failed to read back error");
    return;
  }
  cell_err_[ci]=misfit_[which_dipole]=(*m)[0][0];
  if (!dir_iport_->get(mH) || !mH.get_rep() || mH->ncols()<3) {
    error("DipoleSearch::failed to read back orientation");
    return;
  }
  cell_dir_[ci]=Vector((*m)[0][0], (*m)[0][1], (*m)[0][2]);
  for (j=0; j<3; j++)
    dipoles_(which_dipole,j+3)=(*m)[0][j];
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
    cerr << "Error: seedpoint " <<seed_counter_<<" is outside of mesh!" << endl;
    return 0;
  }
  if (cell_visited_[ci]) {
    cerr << "Warning: redundant seedpoints.\n";
    misfit_[seed_counter_]=cell_err_[ci];
  } else {
    cell_visited_[ci]=1;
    send_and_get_data(seed_counter_, ci);
  }
  seed_counter_++;

  // done seeding, prepare for search phase
  if (seed_counter_ > NSEEDS_) {
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
    if (!cell_visited_[ci]) {
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
}


//! Read the input fields.  Check whether the inputs are valid,
//! and whether they've changed since last time.

void DipoleSearch::read_field_ports(int &valid_data, int &new_data) {
  FieldHandle mesh;
  valid_data=1;
  new_data=0;
  if (mesh_iport_->get(mesh) && mesh.get_rep() &&
      (mesh->get_type_name(0) == "TetVol")) {
    if (!meshH_.get_rep() || (meshH_->generation != mesh->generation)) {
      new_data=1;
      meshH_=mesh;
      // cast the mesh base class up to a tetvolmesh
      vol_mesh_=
	(TetVolMesh*)dynamic_cast<TetVolMesh*>(mesh->mesh().get_rep());
    } else {
      cerr << "DipoleSearch -- same VolumeMesh as before."<<endl;
    }
  } else {
    valid_data=0;
    cerr << "DipoleSearch -- didn't get a valid VolumeMesh."<<endl;
  }
  
  FieldHandle seeds;    
  PointCloud<double> *d;
  if (seeds_iport_->get(seeds) && seeds.get_rep() &&
      (seedsH_->get_type_name(0) == "PointCloud") &&
      (d=dynamic_cast<PointCloud<double> *>(seedsH_.get_rep())) &&
      (d->get_typed_mesh()->nodes_size() == (unsigned int)NSEEDS_)) {
    if (!seedsH_.get_rep() || (seedsH_->generation != seeds->generation)) {
      new_data = 1;
      seedsH_=seeds;
    } else {
      cerr << "DipoleSearch -- same SeedsMesh as before."<<endl;
    }
  } else {
    valid_data=0;
    cerr << "DipoleSearch -- didn't get a valid SeedsMesh."<<endl;
  }
}


//! If we have an old solution and the inputs haven't changed, send that
//! one.  Otherwise, if the input is valid, run a simplex search to find
//! a dipole with minimal misfit.

void DipoleSearch::execute() {
  int valid_data, new_data;
  read_field_ports(valid_data, new_data);
  if (!valid_data) return;
  if (!new_data) {
    if (simplexH_.get_rep()) { // if we have valid old data
      // send old data and clear ports
      simplex_oport_->send(simplexH_);
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
    } else if (state_ == "START_SEARCH") {
      simplex_search();
      state_ = "DONE";
    }
    if (stop_search_ || state_ == "DONE") break;
  }
  if (last_intermediate_) { // last sends were send_intermediates
    // gotta do final sends and clear the ports
    MatrixHandle dummy_mat;
    x_oport_->send(dummy_mat);
    y_oport_->send(dummy_mat);
    z_oport_->send(dummy_mat);
    simplex_oport_->send(simplexH_);
    FieldHandle dummy_fld;
    dipole_oport_->send(dummy_fld);
    misfit_iport_->get(dummy_mat);
    dir_iport_->get(dummy_mat);
  }
  state_ = "SEEDING";
  stop_search_=0;
  seed_counter_=0;
}


//! Commands invoked from the Gui.  Pause/unpause/stop the search.

void DipoleSearch::tcl_command(TCLArgs& args, void* userdata) {
  if (args[1] == "pause") {
    if (mylock_.tryLock())
      cerr << "DipoleSearch pausing..."<<endl;
    else 
      cerr << "DipoleSearch: can't lock -- already locked"<<endl;
  } else if (args[1] == "unpause") {
    if (mylock_.tryLock())
      cerr << "DipoleSearch: can't unlock -- already unlocked"<<endl;
    else
      cerr << "DipoleSearch: unpausing"<<endl;
    mylock_.unlock();
  } else if (args[1] == "stop") {
    stop_search_=1;
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace BioPSE
