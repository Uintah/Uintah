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
 *  OptimizeDipole.cc:  Search for the optimal dipole
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
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <iostream>
using std::endl;
#include <stdio.h>
#include <math.h>


namespace BioPSE {
using namespace SCIRun;

class OptimizeDipole : public Module {
  typedef SCIRun::ConstantBasis<Vector>                           PCVBasis;
  typedef SCIRun::ConstantBasis<double>                           PCDBasis;
  typedef SCIRun::PointCloudMesh<ConstantBasis<Point> >           PCMesh;
  typedef SCIRun::GenericField<PCMesh, PCVBasis, vector<Vector> > PCFieldV;
  typedef SCIRun::GenericField<PCMesh, PCDBasis, vector<double> > PCFieldD;
  typedef TetVolMesh<TetLinearLgn<Point> >                        TVMesh;

  FieldHandle seedsH_;
  FieldHandle meshH_;
  TVMesh::handle_type vol_mesh_;

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
  void send_and_get_data(int which_dipole, TVMesh::Cell::index_type ci);
  int pre_search();
  Vector eval_test_dipole();
  double simplex_step(Array1<double>& sum, double factor, int worst);
  void simplex_search();
  void read_field_ports(int &valid_data, int &new_data);
  void organize_last_send();
  int find_better_neighbor(int best, Array1<double>& sum);
public:
  GuiInt use_cache_gui_;
  OptimizeDipole(GuiContext *context);
  virtual ~OptimizeDipole();
  virtual void execute();
  virtual void tcl_command( GuiArgs&, void * );
};


DECLARE_MAKER(OptimizeDipole)



int OptimizeDipole::NDIM_ = 3;
int OptimizeDipole::NSEEDS_ = 4;
int OptimizeDipole::NDIPOLES_ = 5;
int OptimizeDipole::MAX_EVALS_ = 1001;
double OptimizeDipole::CONVERGENCE_ = 0.00003;
double OptimizeDipole::OUT_OF_BOUNDS_MISFIT_ = 1000000;


OptimizeDipole::OptimizeDipole(GuiContext *context)
  : Module("OptimizeDipole", context, Filter, "Inverse", "BioPSE"),
    meshH_(0),
    mylock_("pause lock for OptimizeDipole"),
    use_cache_gui_(context->subVar("use_cache_gui_"))
{
  mylock_.unlock();
  state_ = "SEEDING";
  stop_search_ = 0;
  seed_counter_ = 0;
}


OptimizeDipole::~OptimizeDipole()
{
}


//! Initialization sets up our dipole search matrix, and misfit vector, and
//!   resizes and initializes our caching vectors

void
OptimizeDipole::initialize_search()
{
  // cast the mesh based class up to a tetvolmesh
  PCMesh *seeds_mesh =
    (PCMesh*)dynamic_cast<PCMesh*>(seedsH_->mesh().get_rep());

  // iterate through the nodes and copy the positions into our
  //  simplex search matrix (`dipoles')
  PCMesh::Node::iterator ni; seeds_mesh->begin(ni);
  PCMesh::Node::iterator nie; seeds_mesh->end(nie);
  misfit_.resize(NDIPOLES_);
  dipoles_.resize(NDIPOLES_, NDIM_+3);
  for (int nc=0; ni != nie; ++ni, nc++)
  {
    Point p;
    seeds_mesh->get_center(p, *ni);
    dipoles_(nc,0)=p.x(); dipoles_(nc,1)=p.y(); dipoles_(nc,2)=p.z();
    dipoles_(nc,3)=0; dipoles_(nc,4)=0; dipoles_(nc,5)=1;
  }
  // this last dipole entry contains our test dipole
  int j;
  for (j=0; j<NDIM_+3; j++)
    dipoles_(NSEEDS_,j)=0;

  TVMesh::Cell::size_type csize;
  vol_mesh_->size(csize);
  cell_visited_.resize(csize);
  cell_err_.resize(csize);
  cell_dir_.resize(csize);
  cell_visited_.initialize(0);
  use_cache_=use_cache_gui_.get();
}


//! Find the misfit and optimal orientation for a single dipole

void
OptimizeDipole::send_and_get_data(int which_dipole, TVMesh::Cell::index_type ci)
{
  if (!mylock_.tryLock())
  {
    msg_stream_ << "Thread is paused\n";
    mylock_.lock();
    mylock_.unlock();
    msg_stream_ << "Thread is unpaused\n";
  }
  else
  {
    mylock_.unlock();
  }

  int j;

  // build columns for MatrixSelectVec
  // each column is an interpolant vector or index/weight pairs
  // we just have one entry for each -- index=leadFieldColumn, weight=1
  
  int *rr = scinew int[4];   rr[0] = 0; rr[1] = 1; rr[2] = 2; rr[3] =3;
  int *cc = scinew int[3];
  double *dd = scinew double[3];
  
  for (int i=0;i<3;i++)
  {
    cc[i] = ci*3+i;
    dd[i] = 1.0;
  }


  TVMesh::Cell::size_type matrix_size;
  vol_mesh_->size(matrix_size);
  matrix_size = matrix_size*3;
  
  
  leadfield_selectH_ = scinew SparseRowMatrix(3,matrix_size,rr,cc,3,dd); 

  PCMesh::handle_type pcm = scinew PCMesh;
  for (j=0; j<NSEEDS_; j++)
    pcm->add_point(Point(dipoles_(j,0), dipoles_(j,1), dipoles_(j,2)));
  PCFieldV *pcv = scinew PCFieldV(pcm);
  for (j=0; j<NSEEDS_; j++)
    pcv->fdata()[j] = Vector(dipoles_(j,3), dipoles_(j,4), dipoles_(j,5));

  pcm = scinew PCMesh;
  pcm->add_point(Point(dipoles_(which_dipole, 0), dipoles_(which_dipole, 1),
		       dipoles_(which_dipole, 2)));
  PCFieldV *pcd = scinew PCFieldV(pcm);

  // send out data
  simplexH_ = pcv;
  dipoleH_ = pcd;
  send_output_handle("LeadFieldSelectionMatrix",
                     leadfield_selectH_, true, true);
  send_output_handle("DipoleSimplex", simplexH_, true, true);
  send_output_handle("TestDipole", dipoleH_, true, true);
  last_intermediate_ = 1;

  // read back data, and set the caches and search matrix
  MatrixHandle mH;
  if (!get_input_handle("TestMisfit", mH, false)) return;

  cell_err_[ci] = misfit_[which_dipole] = mH->get(0, 0);

  if (!get_input_handle("TestDirection", mH, false)) return;
  if (mH->nrows() < 3)
  {
    error("TestDirection had wrong size (not 3 rows).");
    return;
  }
  cell_dir_[ci] = Vector(mH->get(0, 0), mH->get(1, 0), mH->get(2, 0));
  for (j=0; j<3; j++)
    dipoles_(which_dipole,j+3) = mH->get(j, 0);
}


//! pre_search gets called once for each seed dipole.  It sends out
//!   one of the seeds, reads back the results, and fills the data
//!   into the caches and the search matrix
//! return "fail" if any seeds are out of mesh or if we don't get
//!   back a misfit or optimal orientation after a send

int
OptimizeDipole::pre_search()
{
  if (seed_counter_ == 0) {
    initialize_search();
  }

  // send out a seed dipole and get back the misfit and the optimal orientation
  TVMesh::Cell::index_type ci;
  if (!vol_mesh_->locate(ci, Point(dipoles_(seed_counter_,0),
				   dipoles_(seed_counter_,1),
				   dipoles_(seed_counter_,2))))
  {
    warning("Seedpoint " + to_string(seed_counter_) + " is outside of mesh.");
    return 0;
  }
  if (cell_visited_[ci])
  {
    warning("Redundant seedpoints found.");
    misfit_[seed_counter_]=cell_err_[ci];
  }
  else
  {
    cell_visited_[ci]=1;
    send_and_get_data(seed_counter_, ci);
  }
  seed_counter_++;

  // done seeding, prepare for search phase
  if (seed_counter_ == NSEEDS_)
  {
    seed_counter_ = 0;
    state_ = "START_SEARCHING";
  }
  return 1;
}


//! Evaluate a test dipole.  Return the optimal orientation.

Vector
OptimizeDipole::eval_test_dipole()
{
  TVMesh::Cell::index_type ci;
  if (vol_mesh_->locate(ci, Point(dipoles_(NSEEDS_,0),
				  dipoles_(NSEEDS_,1),
				  dipoles_(NSEEDS_,2))))
  {
    if (!cell_visited_[ci] || !use_cache_)
    {
      cell_visited_[ci]=1;
      send_and_get_data(NSEEDS_, ci);
    }
    else
    {
      misfit_[NSEEDS_]=cell_err_[ci];
    }
    return (cell_dir_[ci]);
  }
  else
  {
    misfit_[NSEEDS_]=OUT_OF_BOUNDS_MISFIT_;
    return (Vector(0,0,1));
  }
}


//! Check to see if any of the neighbors of the "best" dipole are better.

int
OptimizeDipole::find_better_neighbor(int best, Array1<double>& sum)
{
  Point p(dipoles_(best, 0), dipoles_(best, 1), dipoles_(best, 2));
  TVMesh::Cell::index_type ci;
  vol_mesh_->locate(ci, p);
  TVMesh::Cell::array_type ca;
  vol_mesh_->synchronize(Mesh::FACE_NEIGHBORS_E);
  vol_mesh_->get_neighbors(ca, ci);
  for (unsigned int n=0; n<ca.size(); n++)
  {
    Point p1;
    vol_mesh_->get_center(p1, ca[n]);
    dipoles_(NSEEDS_,0)=p1.x();
    dipoles_(NSEEDS_,1)=p1.y();
    dipoles_(NSEEDS_,2)=p1.z();
    eval_test_dipole();
    if (misfit_[NSEEDS_] < misfit_[best])
    {
      misfit_[best] = misfit_[NSEEDS_];
      int i;
      for (i=0; i<NDIM_; i++)
      {
        sum[i] = sum[i] + dipoles_(NSEEDS_,i)-dipoles_(best,i);
        dipoles_(best,i) = dipoles_(NSEEDS_,i);
      }
      for (i=NDIM_; i<NDIM_+3; i++)
      {
        dipoles_(best,i) = dipoles_(NSEEDS_,i);  // copy orientation data
      }
    }
    return 1;
  }
  return 0;
}


//! Take a single simplex step.  Evaluate a new position -- if it's
//! better then an existing vertex, swap them.

double
OptimizeDipole::simplex_step(Array1<double>& sum, double factor, int worst)
{
  double factor1 = (1 - factor)/NDIM_;
  double factor2 = factor1-factor;
  int i;
  for (i=0; i<NDIM_; i++)
  {
    dipoles_(NSEEDS_,i) = sum[i]*factor1 - dipoles_(worst,i)*factor2;
  }

  // evaluate the new guess
  eval_test_dipole();

  // if this is better, swap it with the worst one
  if (misfit_[NSEEDS_] < misfit_[worst])
  {
    misfit_[worst] = misfit_[NSEEDS_];
    for (i=0; i<NDIM_; i++)
    {
      sum[i] = sum[i] + dipoles_(NSEEDS_,i)-dipoles_(worst,i);
      dipoles_(worst,i) = dipoles_(NSEEDS_,i);
    }
    for (i=NDIM_; i<NDIM_+3; i++)
      dipoles_(worst,i) = dipoles_(NSEEDS_,i);  // copy orientation data
  }

  return misfit_[NSEEDS_];
}


//! The simplex has been constructed -- now let's search for a minimal misfit

void
OptimizeDipole::simplex_search()
{
  Array1<double> sum(NDIM_);  // sum of the entries in the search matrix rows
  sum.initialize(0);
  int i, j;
  for (i=0; i<NSEEDS_; i++)
    for (j=0; j<NDIM_; j++)
      sum[j]+=dipoles_(i,j);

  double relative_tolerance = 0;
  int num_evals = 0;

  for( ;; )
  {
    int best, worst, next_worst;
    best = 0;
    if (misfit_[0] > misfit_[1])
    {
      worst = 0;
      next_worst = 1;
    }
    else
    {
      worst = 1;
      next_worst = 0;
    }
    int i;
    for (i=0; i<NSEEDS_; i++)
    {
      if (misfit_[i] <= misfit_[best]) best=i;
      if (misfit_[i] > misfit_[worst])
      {
        next_worst = worst;
        worst = i;
      }
      else if (misfit_[i] > misfit_[next_worst] && (i != worst))
      {
        next_worst=i;
      }
      relative_tolerance = 2*(misfit_[worst]-misfit_[best])/
        (misfit_[worst]+misfit_[best]);
    }

    if (num_evals > MAX_EVALS_ || stop_search_) break;

    // make sure all of our neighbors are worse than us...
    if (relative_tolerance < CONVERGENCE_)
    {
      if (misfit_[best]>1.e-12 && find_better_neighbor(best, sum))
      {
        num_evals++;
        continue;
      }
      break;
    }

    double step_misfit = simplex_step(sum, -1, worst);
    num_evals++;
    if (step_misfit <= misfit_[best])
    {
      step_misfit = simplex_step(sum, 2, worst);
      num_evals++;
    }
    else if (step_misfit >= misfit_[worst])
    {
      double old_misfit = misfit_[worst];
      step_misfit = simplex_step(sum, 0.5, worst);
      num_evals++;
      if (step_misfit >= old_misfit)
      {
        for (i=0; i<NSEEDS_; i++)
        {
          if (i != best)
          {
            int j;
            for (j=0; j<NDIM_; j++)
                  {
              dipoles_(i,j) = dipoles_(NSEEDS_,j) =
          0.5 * (dipoles_(i,j) + dipoles_(best,j));
              dipoles_(NSEEDS_,j)=dipoles_(i,j);
            }
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
      for (i=0; i<NSEEDS_; i++)
      {
        for (j=0; j<NDIM_; j++)
        {
          sum[j]+=dipoles_(i,j);
        }
      }
    }
  }
  msg_stream_ << "OptimizeDipole -- num_evals = "<<num_evals << "\n";
}


//! Read the input fields.  Check whether the inputs are valid,
//! and whether they've changed since last time.

void
OptimizeDipole::read_field_ports(int &valid_data, int &new_data)
{
  valid_data = 1;
  new_data = 0;
  FieldHandle mesh(0);
  if (get_input_handle("TetMesh", mesh, false))
  {
    const string &mtdn = mesh->mesh()->get_type_description()->get_name();
    if (mtdn == get_type_description((TVMesh*)0)->get_name())
    {
      if (!meshH_.get_rep() || (meshH_->generation != mesh->generation))
      {
        new_data = 1;
        meshH_ = mesh;

        // Cast the mesh base class up to a tetvolmesh.
        vol_mesh_ = (TVMesh*)dynamic_cast<TVMesh*>(mesh->mesh().get_rep());
        vol_mesh_->synchronize(Mesh::LOCATE_E);
      }
      else
      {
        remark("Same VolumeMesh as previous run.");
      }
    }
    else
    {
      valid_data = 0;
      remark("Didn't get a valid VolumeMesh.");
    }
  }
  else
  {
    valid_data = 0;
    error("No input available on the mesh field port.");
  }
  
  FieldHandle seeds;
  if (!get_input_handle("DipoleSeeds", seeds, false))
  {
    warning("No input seeds.");
    valid_data = 0;
  }
  else
  {
    PCFieldD *d = dynamic_cast<PCFieldD*>(seeds.get_rep());
    if (d == NULL)
    {
      warning("Seeds typename should have been PointCloudField<double>.");
      valid_data = 0;
    }
    else
    {
      PCMesh::Node::size_type nsize; d->get_typed_mesh()->size(nsize);
      if (nsize != (unsigned int)NSEEDS_)
      {
        msg_stream_ << "Got "<< nsize <<" seeds, instead of "<<NSEEDS_<<"\n";
        valid_data=0;
      }
      else if (!seedsH_.get_rep() || (seedsH_->generation != seeds->generation))
      {
        new_data = 1;
        seedsH_ = seeds;
      }
      else
      {
        remark("Using same seeds as before.");
      }
    }
  }
}


void
OptimizeDipole::organize_last_send()
{
  int bestIdx=0;
  double bestMisfit = misfit_[0];
  int i;
  for (i=1; i<misfit_.size(); i++)
  {
    if (misfit_[i] < bestMisfit)
    {
      bestMisfit = misfit_[i];
      bestIdx = i;
    }
  }

  Point best_pt(dipoles_(bestIdx,0), dipoles_(bestIdx,1), dipoles_(bestIdx,2));
  TVMesh::Cell::index_type best_cell_idx;
  vol_mesh_->locate(best_cell_idx, best_pt);
  TVMesh::Cell::size_type matrix_size;
  vol_mesh_->size(matrix_size);
  matrix_size = matrix_size*3;
  
  msg_stream_ << "OptimizeDipole -- the dipole was found in cell " << 
    best_cell_idx << "\n    at position " << best_pt << 
    " with a misfit of " << bestMisfit << "\n";

  int *rr = scinew int[4];   rr[0] = 0; rr[1] = 1; rr[2] = 2; rr[3] =3;
  int *cc = scinew int[3];
  double *dd = scinew double[3];
  
  for (int i=0;i<3;i++)
  {
    cc[i] = best_cell_idx*3+i;
    dd[i] = 1.0;
  }

  leadfield_selectH_ = scinew SparseRowMatrix(3,matrix_size,rr,cc,3,dd);
  PCMesh::handle_type pcm = scinew PCMesh;
  pcm->add_point(best_pt);
  PCFieldV *pcd = scinew PCFieldV(pcm);
  pcd->fdata()[0]=Vector(dipoles_(bestIdx,3), dipoles_(bestIdx,4), dipoles_(bestIdx,5));
  dipoleH_ = pcd;
}


//! If we have an old solution and the inputs haven't changed, send that
//! one.  Otherwise, if the input is valid, run a simplex search to find
//! a dipole with minimal misfit.

void
OptimizeDipole::execute()
{
  int valid_data, new_data;

  read_field_ports(valid_data, new_data);
  if (!valid_data) return;

  if (!new_data)
  {
    if (simplexH_.get_rep())
    { // if we have valid old data
      // send old data and clear ports
      send_output_handle("LeadFieldSelectionMatrix", leadfield_selectH_, true);
      send_output_handle("DipoleSimplex", simplexH_, true);
      send_output_handle("TestDipole", dipoleH_, true);

      MatrixHandle dummy_mat;
      get_input_handle("TestMisfit", dummy_mat, false);
     get_input_handle("TestDirection", dummy_mat, false);
      return;
    }
    else
    {
      return;
    }
  }

  last_intermediate_=0;

  // we have new, valid data -- run the simplex search
  for( ;; )
  {
    if (state_ == "SEEDING")
    {
      if (!pre_search()) break;
    }
    else if (state_ == "START_SEARCHING")
    {
      simplex_search();
      state_ = "DONE";
    }
    if (stop_search_ || state_ == "DONE") break;
  }
  
  if (last_intermediate_)
  { // last sends were send_intermediates
    // gotta do final sends and clear the ports
    organize_last_send();
    send_output_handle("LeadFieldSelectionMatrix", leadfield_selectH_, true);
    send_output_handle("DipoleSimplex", simplexH_, true);
    send_output_handle("TestDipole", dipoleH_, true);

    MatrixHandle dummy_mat;
    get_input_handle("TestMisfit", dummy_mat, false);
    get_input_handle("TestDirection", dummy_mat, false);
  }
  state_ = "SEEDING";
  stop_search_=0;
  seed_counter_=0;
}


//! Commands invoked from the Gui.  Pause/unpause/stop the search.

void
OptimizeDipole::tcl_command(GuiArgs& args, void* userdata)
{
  if (args[1] == "pause")
  {
    if (mylock_.tryLock())
    {
      msg_stream_ << "TCL initiating pause..."<<endl;
    }
    else
    {
      msg_stream_ << "Can't lock -- already locked"<<endl;
    }
  }
  else if (args[1] == "unpause")
  {
    if (mylock_.tryLock())
    {
      msg_stream_ << "Can't unlock -- already unlocked"<<endl;
    }
    else
    {
      msg_stream_ << "TCL initiating unpause..."<<endl;
    }
    mylock_.unlock();
  }
  else if (args[1] == "stop")
  {
    stop_search_=1;
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace BioPSE
