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
#include <Core/Containers/String.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Math/MusilRNG.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <math.h>

#define NDIM 3
#define NSEEDS (NDIM+1)
#define NDIPOLES (NDIM+2)
#define CONVERGANCE 0.001
#define MAXEVALS 100

namespace BioPSE {
using namespace SCIRun;

class DipoleSearch : public Module {    
  FieldIPort     *seeds_iport;
  FieldIPort     *mesh_iport;
  MatrixIPort    *misfit_iport;
  MatrixIPort    *dir_iport;

  MatrixOPort    *x_oport;
  MatrixOPort    *y_oport;
  MatrixOPort    *z_oport;
  FieldOPort     *simplex_oport;
  FieldOPort     *dipole_oport;

  FieldHandle seedsH;
  FieldHandle meshH;
  TetVolMeshHandle vol_mesh;
  FieldHandle simplexH;

  int seed_counter;
  clString state;
  Array1<double> errors;
  Array2<double> dipoles;
  Array1<int> cell_visited;
  Array1<double> cell_err;
  Array1<Vector> cell_dir;  
  int use_cache;
  int in_bounds;
  int stop_search;
  int send_intermediate;  // was the last send we did a send_intermediate
  Mutex mylock;

  // private methods
  void initialize_search();
  void send_and_get_data(int which_dipole, Point p, TetVolMesh::cell_index ci,
			 Vector &dir);
  int pre_search();
  void eval_test_dipole(Vector &dir);
  double simplex_step(Array1<double>& sum, double factor, int worst);
  void simplex_search();
  void read_field_ports(int &valid_data, int &new_data);
public:
  GuiString tcl_status;
  GuiInt useCacheTCL;
  DipoleSearch(const clString& id);
  virtual ~DipoleSearch();
  virtual void execute();
  void tcl_command( TCLArgs&, void * );
};

extern "C" Module* make_DipoleSearch(const clString& id) {
  return new DipoleSearch(id);
}

DipoleSearch::DipoleSearch(const clString& id)
  : Module("DipoleSearch", id, Filter), tcl_status("tcl_status",id,this),
    mylock("pause lock for DipoleSearch"), useCacheTCL("useCacheTCL",id,this)
{
  // point cloud of vectors -- the seed positions/orientations for our search
  seeds_iport = new FieldIPort(this, "DipoleSeeds",
				      FieldIPort::Atomic);
  add_iport(seeds_iport);
  
  // domain of search -- used for constraining and for caching misfits
  mesh_iport = new FieldIPort(this,"TetMesh",
			      FieldIPort::Atomic);
  add_iport(mesh_iport);

  // the computed misfit for the latest test dipole
  misfit_iport = new MatrixIPort(this, "TestMisfit",
				 MatrixIPort::Atomic);
  add_iport(misfit_iport);
  
  // optimal orientation for the test dipole
  dir_iport = new MatrixIPort(this,"TestDirection",
			      MatrixIPort::Atomic);
  add_iport(dir_iport);
  
  // column matrix of the position and x-orientation of the test dipole
  x_oport = new MatrixOPort(this, "TestDipoleX",
				MatrixIPort::Atomic);
  add_oport(x_oport);
  
  // column matrix of the position and y-orientation of the test dipole
  y_oport = new MatrixOPort(this, "TestDipoleY",
				MatrixIPort::Atomic);
  add_oport(y_oport);
  
  // column matrix of the position and z-orientation of the test dipole
  z_oport = new MatrixOPort(this, "TestDipoleZ",
				MatrixIPort::Atomic);
  add_oport(z_oport);
  
  // point cloud of vectors -- the latest simplex (for vis)
  simplex_oport = new FieldOPort(this, "DipoleSimplex",
				FieldIPort::Atomic);
  add_oport(simplex_oport);

  // point cloud of one vector, just the test dipole (for vis)
  dipole_oport = new FieldOPort(this, "TestDipole",
				FieldIPort::Atomic);
  add_oport(dipole_oport);

  mylock.unlock();
  state = "SEEDING";
  stop_search = 0;
  seed_counter = 0;
}

DipoleSearch::~DipoleSearch(){}

//! Initialization reads and validates the volume and seed meshes;
//!   sets up our dipole search matrix; builds the map we'll use
//!   to keep track of which seed dipole is where in the search
//!   matrix; and resizes and initializes our caching vectors

void DipoleSearch::initialize_search() {
  // cast the mesh based calss up to a tetvolmesh
  TetVolMesh *seeds_mesh =
    (TetVolMesh*)dynamic_cast<TetVolMesh*>(seedsH->mesh().get_rep());

  // iterate through the nodes and copy the positions into our 
  //  simplex search matrix (`dipoles')
  TetVolMesh::node_iterator ni = seeds_mesh->node_begin();
  errors.resize(NDIM);
  dipoles.newsize(NDIPOLES, NDIM+3);
  for (int nc=0; ni != seeds_mesh->node_end(); ++ni, nc++) {
    Point p;
    seeds_mesh->get_center(p, *ni);
    dipoles(nc,0)=p.x(); dipoles(nc,1)=p.y(); dipoles(nc,2)=p.z();
    dipoles(nc,3)=0; dipoles(nc,4)=0; dipoles(nc,5)=1;
  }
  // this last dipole entry contains our test dipole
  int j;
  for (j=0; j<NDIM+3; j++) 
    dipoles(NSEEDS,j)=0;

  cell_visited.resize(vol_mesh->cells_size());
  cell_err.resize(vol_mesh->cells_size());
  cell_dir.resize(vol_mesh->cells_size());
  cell_visited.initialize(0);
  use_cache=useCacheTCL.get();
}

void DipoleSearch::send_and_get_data(int which_dipole, Point p, 
				     TetVolMesh::cell_index ci,
				     Vector &dir) {
  if (!mylock.tryLock()) {
    mylock.lock();
    mylock.unlock();
  } else {
    mylock.unlock();
  }

  ColumnMatrix *x_out = scinew ColumnMatrix(7);
  ColumnMatrix *y_out = scinew ColumnMatrix(7);
  ColumnMatrix *z_out = scinew ColumnMatrix(7);
  int j;
  for (j=0; j<3; j++)
    (*x_out)[j] = (*y_out)[j] = (*z_out)[j] = dipoles(seed_counter,j);
  for (j=3; j<6; j++)
    (*x_out)[j] = (*y_out)[j] = (*z_out)[j] = 0;
  (*x_out)[3] = (*y_out)[4] = (*z_out)[5] = 1;
  (*x_out)[6] = (*y_out)[6] = (*z_out)[6] = (double)ci;
  
  TetVolMeshHandle tvm = scinew TetVolMesh;
  for (j=0; j<NSEEDS; j++)
    tvm->add_point(Point(dipoles(j,0), dipoles(j,1), dipoles(j,2)));
  TetVol<Vector> *tvv = scinew TetVol<Vector>(tvm, Field::NODE);
  for (j=0; j<NSEEDS; j++)
    tvv->fdata()[j] = Vector(dipoles(j,3), dipoles(j,4), dipoles(j,5));

  tvm = scinew TetVolMesh;
  tvm->add_point(p);
  TetVol<double> *tvd = scinew TetVol<double>(tvm, Field::NODE);
  
  // send out data
  x_oport->send_intermediate(x_out);
  y_oport->send_intermediate(y_out);
  z_oport->send_intermediate(z_out);
  simplexH = tvv;
  simplex_oport->send_intermediate(simplexH);
  dipole_oport->send_intermediate(tvd);
  send_intermediate=1;

  // read back data, and set the caches and search matrix
  MatrixHandle mH;
  Matrix* m;
  if (!misfit_iport->get(mH) || !(m = mH.get_rep())) {
    error("DipoleSearch::preSearch failed to read back error");
    return;
  }
  cell_err[ci]=errors[which_dipole]=(*m)[0][0];
  if (!dir_iport->get(mH) || !mH.get_rep() || mH->ncols()<3) {
    error("DipoleSearch::preSearch failed to read back orientation");
    return;
  }
  dir=cell_dir[ci]=Vector((*m)[0][0], (*m)[0][1], (*m)[0][2]);
  for (j=0; j<3; j++)
    dipoles(which_dipole,j+3)=(*m)[0][j];
}  



//! pre_search gets called once for each seed dipole.  It sends out
//!   one of the seeds, reads back the results, and fills the data
//!   into the caches and the search matrix
//! return "fail" if any seeds are out of mesh or if we don't get
//!   back a misfit or optimal orientation after a send 

int DipoleSearch::pre_search() {
  if (seed_counter == 0) {
    initialize_search();
  }

  // send out a seed dipole and get back the error and the optimal orientation
  TetVolMesh::cell_index ci;
  Point p(dipoles(seed_counter,0), dipoles(seed_counter,1), 
	  dipoles(seed_counter,2));
  if (!vol_mesh->locate(ci, p)) {
    cerr << "Error: seedpoint " <<seed_counter<<" is outside of mesh!" << endl;
    return 0;
  }
  if (cell_visited[ci]) {
    cerr << "Warning: redundant seedpoints.\n";
  } else {
    cell_visited[ci]=1;
    Vector dir;
    send_and_get_data(seed_counter, p, ci, dir);
  }
  seed_counter++;

  // done seeding, prepare for search phase
  if (seed_counter > NSEEDS) {
    seed_counter = 0;
    state = "START_SEARCHING";
  }
  return 1;
}

// evaluate the test dipole, cache the misfit and optimal orientation
void DipoleSearch::eval_test_dipole(Vector &dir) {
  TetVolMesh::cell_index ci;
  Point p(dipoles(NSEEDS,0), dipoles(NSEEDS,1), dipoles(NSEEDS,2));
  if (vol_mesh->locate(ci, p)) {
    if (!cell_visited[ci]) {
      cell_visited[ci]=1;
      send_and_get_data(NSEEDS, p, ci, dir);
    } else {
      dir=cell_dir[ci];
      errors[NSEEDS]=cell_err[ci];
    }
  }
}

double DipoleSearch::simplex_step(Array1<double>& sum, double factor,
				  int worst) {
  double factor1 = (1 - factor)/NDIM;
  double factor2 = factor1-factor;
  int i;
  for (i=0; i<NDIM; i++) 
    dipoles(NSEEDS,i) = sum[i]*factor1 - dipoles(worst,i)*factor2;

  // evaluate the new guess
  Vector dir(0,0,1);
  eval_test_dipole(dir);

  // if this is better, swap it with the worst one
  if (errors[NSEEDS] < errors[worst]) {
    errors[worst] = errors[NSEEDS];
    for (i=0; i<NDIM; i++) {
      sum[i] = sum[i] + dipoles(NSEEDS,i)-dipoles(worst,i);
      dipoles(worst,i) = dipoles(NSEEDS,i);
    }
    for (i=NDIM; i<NDIM+3; i++)
      dipoles(worst,i) = dipoles(NSEEDS,i);  // copy orientation data
  }

  return errors[NSEEDS];
}


//! the simplex has been constructed -- now let's find a minimum
void DipoleSearch::simplex_search() {
  Array1<double> sum(NDIM);  // sum of the entries in the search matrix rows
  sum.initialize(0);
  int i, j;
  for (i=0; i<NSEEDS; i++) 
    for (j=0; j<NDIM; j++)
      sum[i]+=dipoles(i,j); 

  double relative_tolerance;
  int num_evals = 0;

  while(1) {
    int best, worst, next_worst;
    best = 0;
    if (errors[0] > errors[1]) {
      worst = 0;
      next_worst = 1;
    } else {
      worst = 1;
      next_worst = 0;
    }
    int i;
    for (i=0; i<NSEEDS; i++) {
      if (errors[i] <= errors[best]) best=i;
      if (errors[i] > errors[worst]) {
	next_worst = worst;
	worst = i;
      } else 
	if (errors[i] > errors[next_worst] && (i != worst)) 
	  next_worst=i;
      relative_tolerance = 2*(errors[worst]-errors[best])/
	(errors[worst]+errors[best]);
    }

    if ((relative_tolerance < CONVERGANCE) || 
	(num_evals > MAXEVALS) || (stop_search)) 
      break;

    double step_error = simplex_step(sum, -1, worst);
    num_evals++;
    if (step_error <= errors[best]) {
      step_error = simplex_step(sum, 2, worst);
      num_evals++;
    } else if (step_error >= errors[worst]) {
      double old_error = errors[worst];
      step_error = simplex_step(sum, 0.5, worst);
      num_evals++;
      if (step_error >= old_error) {
	for (i=0; i<NSEEDS; i++) {
	  if (i != best) {
	    int j;
	    for (j=0; j<NDIM; j++)
	      dipoles(i,j) = dipoles(NSEEDS,j) = 
		0.5 * (dipoles(i,j) + dipoles(best,j));
	    Vector dir(0,0,1);
	    eval_test_dipole(dir);
	    errors[i] = errors[NDIM];
	    num_evals++;
	    dipoles(i,3) = dir.x(); 
	    dipoles(i,4) = dir.y(); 
	    dipoles(i,5) = dir.z(); 
	  }
	}
      }
      sum.initialize(0);
      for (i=0; i<NSEEDS; i++) {
	for (j=0; j<NDIM; j++)
	  sum[i]+=dipoles(i,j); 
      }
    }
  }
}

void DipoleSearch::read_field_ports(int &valid_data, int &new_data) {
  FieldHandle mesh;
  valid_data=1;
  new_data=0;
  if (mesh_iport->get(mesh) && mesh.get_rep() &&
      (meshH->get_type_name(0) == "TetVol")) {
    if (!meshH.get_rep() || (meshH->generation != mesh->generation)) {
      new_data=1;
      meshH=mesh;
      // cast the mesh base class up to a tetvolmesh
      vol_mesh=(TetVolMesh*)dynamic_cast<TetVolMesh*>(meshH->mesh().get_rep());
    } else {
      cerr << "DipoleSearch -- same VolumeMesh as before."<<endl;
    }
  } else {
    valid_data=0;
    cerr << "DipoleSearch -- didn't get a valid VolumeMesh."<<endl;
  }
  
  FieldHandle seeds;    
  TetVol<double> *d;
  if (seeds_iport->get(seeds) && seeds.get_rep() &&
      (seedsH->get_type_name(0) == "TetVol") &&
      (d=dynamic_cast<TetVol<double> *>(seedsH.get_rep())) &&
      (d->get_typed_mesh()->nodes_size() == NSEEDS)) {
    if (!seedsH.get_rep() || (seedsH->generation != seeds->generation)) {
      new_data = 1;
      seedsH=seeds;
    } else {
      cerr << "DipoleSearch -- same SeedsMesh as before."<<endl;
    }
  } else {
    valid_data=0;
    cerr << "DipoleSearch -- didn't get a valid SeedsMesh."<<endl;
  }
}

void DipoleSearch::execute() {
  int valid_data, new_data;
  read_field_ports(valid_data, new_data);
  if (!valid_data) return;
  if (!new_data) {
    if (simplexH.get_rep()) { // if we have valid old data
      // send old data and clear ports
      simplex_oport->send(simplexH);
      MatrixHandle dummy_mat;
      misfit_iport->get(dummy_mat);
      dir_iport->get(dummy_mat);
      return;
    } else {
      return;
    }
  }

  send_intermediate=0;

  // we have new, valid data -- run the simplex search
  while (1) {
    if (state == "SEEDING") {
      if (!pre_search()) break;
    } else if (state == "START_SEARCH") {
      simplex_search();
      state = "DONE";
    }
    if (stop_search || state == "DONE") break;
  }
  if (send_intermediate) { // last sends were send_intermediates
    // gotta do final sends and clear the ports
    MatrixHandle dummy_mat;
    x_oport->send(dummy_mat);
    y_oport->send(dummy_mat);
    z_oport->send(dummy_mat);
    simplex_oport->send(simplexH);
    FieldHandle dummy_fld;
    dipole_oport->send(dummy_fld);
    misfit_iport->get(dummy_mat);
    dir_iport->get(dummy_mat);
  }
  state = "SEEDING";
  stop_search=0;
  seed_counter=0;
}

void DipoleSearch::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "pause") {
        if (mylock.tryLock())
	  cerr << "DipoleSearch pausing..."<<endl;
        else 
	  cerr << "DipoleSearch: can't lock -- already locked"<<endl;
    } else if (args[1] == "unpause") {
        if (mylock.tryLock())
	  cerr << "DipoleSearch: can't unlock -- already unlocked"<<endl;
        else
	  cerr << "DipoleSearch: unpausing"<<endl;
        mylock.unlock();
    } else if (args[1] == "stop") {
        stop_search=1;
    } else {
        Module::tcl_command(args, userdata);
    }
}

} // End namespace BioPSE
