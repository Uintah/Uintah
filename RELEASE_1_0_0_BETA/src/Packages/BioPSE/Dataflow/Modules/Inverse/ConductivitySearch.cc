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
 *  ConductivitySearch.cc:  Solve for the optimal conductivities for a mesh
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Institute
 *
 */

#include <Packages/DaveW/ThirdParty/NumRec/amoeba.h>
#include <Packages/DaveW/ThirdParty/NumRec/nrutil.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Math/MusilRNG.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <math.h>

namespace DaveW {
using namespace SCIRun;


// these static variables are necessary b/c NumRec needs a pointer
// to a static function (error_eval) in Amoeba.  the variables
// referenced in that function, therefore, need to be static.  ugh.
    
double ** _CS_p; //holds initial dipole configuration 
int _CS_nCondReg;
int _CS_in_bounds;
int _CS_need_error;
int _CS_send_pos;
int _CS_stop;
MatrixHandle _CS_cond_params;
double * _CS_err;
Semaphore * _CS_error_sem, * _CS_pos_sem, *_CS_helper_sem;

class ConductivitySearch : public Module {    
  MeshIPort* mesh_iport;
  MatrixIPort* cond_iport;
  MatrixIPort* mat_iport;
  ColumnMatrixIPort* rms_iport;
  MatrixOPort* mat_oport;
  ColumnMatrixOPort* cond_oport;
  double FTOL;
  int nfunc;
  int counter;
  clString state;
  double* y; //holds initial errors for p configuration
  GuiString seedTCL;
public:
  MeshHandle mesh;
  MusilRNG* mr;
  MatrixHandle AmatH;   // hold the stiffness matrix - we want its shape
  SparseRowMatrix *AmatHp;
  int AHgen;
  int seed;
  ColumnMatrixHandle conds;

  Array1<Array1<double> > dataC;
  Array1<Array1<int> > nzeros;
  
  int pinzero, refnode;
  Mutex mylock;
  GuiString refnodeTCL;
  GuiInt pinzeroTCL;
  ConductivitySearch(const clString& id);
  virtual ~ConductivitySearch();
  void buildCompositeMat(const Array1<double> &conds);
  void buildCondMatrices(int);
  virtual void execute();
  double gaussian(double w);
  static double *error_eval(int);
  void helper(int proc);
  void tcl_command( TCLArgs&, void * );
}; //class


extern "C" Module* make_ConductivitySearch(const clString& id) {
  return new ConductivitySearch(id);
}

//---------------------------------------------------------------
ConductivitySearch::ConductivitySearch(const clString& id)
  : Module("ConductivitySearch", id, Filter), 
  mylock("pause lock for ConductivitySearch"),
  pinzeroTCL("pinzeroTCL", id, this), refnodeTCL("refnodeTCL", id, this),
  seedTCL("seedTCL", id, this)
{
  mesh_iport = new MeshIPort(this,"Mesh",
			     MeshIPort::Atomic);
  add_iport(mesh_iport);
    
  cond_iport = new MatrixIPort(this, "Conductivity Parameters",
			       MatrixIPort::Atomic);
  add_iport(cond_iport);
    
  mat_iport = new MatrixIPort(this, "Stiffness Matrix",
			      MatrixIPort::Atomic);
  add_iport(mat_iport);
    
  rms_iport = new ColumnMatrixIPort(this, "RMS",
				    ColumnMatrixIPort::Atomic);
  add_iport(rms_iport);
    
  mat_oport = new MatrixOPort(this,"A Matrix",
			      MatrixIPort::Atomic);
  add_oport(mat_oport);
    
  cond_oport = new ColumnMatrixOPort(this, "Old and New Conductivities", 
				     ColumnMatrixIPort::Atomic);
  add_oport(cond_oport);

  counter = 0;
  state = "START";
    
  _CS_error_sem = new Semaphore("ConductivitySearch error sync", 0);
  _CS_pos_sem = new Semaphore("ConductivitySearch position sync", 0);
  _CS_helper_sem = new Semaphore("ConductivitySearch amoeba sync", 0);
    
  Thread::parallel(Parallel<ConductivitySearch>(this, &ConductivitySearch::helper), 1, false);
  //    Task::multiprocess(1, start_me_up, this, false);
    
  cerr<<"Constructor Done!"<<endl;
  mylock.unlock();
  AHgen=-1;
}

//------------------------------------------------------------
ConductivitySearch::~ConductivitySearch(){}

//--------------------------------------------------------------

double ConductivitySearch::gaussian(double sigma) {
  double x;
  x = 2.0 * (*mr)() - 1.0;
  return x*sigma*sqrt((-2.0 * log(x*x)) / (x*x));
}

void ConductivitySearch::buildCompositeMat(const Array1<double> &conds) {
  int c, idx, nz;
  double *data=AmatHp->a;
  int nnz=AmatHp->nnz;
  for (idx=0; idx<nnz; idx++) 
    data[idx]=0;
  for (c=0; c<_CS_nCondReg; c++)
    for (nz=0; nz<nzeros[c].size(); nz++) {
      idx=nzeros[c][nz];
      data[idx] += dataC[c][idx]*conds[c];
    }

  cerr << "Building composite: conds = ";
  for (c=0; c<conds.size(); c++)
    cerr << conds[c]<<" ";
  cerr << "\n";

#if 0  
  for (c=0; c<conds.size(); c++) {
    cerr << "nzeros["<<c<<"] ";
    for (nz=0; nz<nzeros[c].size(); nz++) {
      cerr << "("<<nz<<") "<< nzeros[c][nz]<<"  ";
    }
    cerr << "\n";
    cerr << "dataC["<<c<<"] ";
    for (idx=0; idx<dataC[c].size(); idx++) {
      cerr << "("<<idx<<") "<<dataC[c][idx]<<"  ";
    }
    cerr << "\n";
  }

  cerr << "data: ";
  for (idx=0; idx<dataC[0].size(); idx++) {
    cerr << "("<<idx<<") "<<data[idx]<<"  ";
  }
  cerr << "\n";
#endif
}

void ConductivitySearch::buildCondMatrices(int ncond) {
  if (dataC.size() != ncond) {
    dataC.resize(ncond);
    nzeros.resize(ncond);
  }
  int i,j,c,e;
  for (c=0; c<ncond; c++) {
    dataC[c].resize(AmatHp->nnz);
    dataC[c].initialize(0);
    nzeros[c].resize(0);
  }
  
  // go through the mesh and build all of the nonzeros for each element into
  // dataC and nzeros
  
  // for each element
  //   build the local matrix (w/o cond) and store non-zeros in dataC[i]
  // for each conductivity type c
  //   for each nonzero in dataC[c]
  //     add its index to nzeros[c]

  for (e=0; e<mesh->elems.size(); e++){

    // build local matrix
    double lcl_a[4][4];
    for(i=0;i<4;i++)
      for(j=0;j<4;j++)
	lcl_a[i][j]=0;

    Element *elem=mesh->elems[e];
    Point pt;
    Vector grad1,grad2,grad3,grad4;
    double vol = mesh->get_grad(elem,pt,grad1,grad2,grad3,grad4);
    if(vol < 1.e-10) {
      cerr << "Skipping element..., volume=" << vol << endl;
      break;
    }
   
    double el_coefs[4][3];
    el_coefs[0][0]=grad1.x();el_coefs[0][1]=grad1.y();el_coefs[0][2]=grad1.z();
    el_coefs[1][0]=grad2.x();el_coefs[1][1]=grad2.y();el_coefs[1][2]=grad2.z();
    el_coefs[2][0]=grad3.x();el_coefs[2][1]=grad3.y();el_coefs[2][2]=grad3.z();
    el_coefs[3][0]=grad4.x();el_coefs[3][1]=grad4.y();el_coefs[3][2]=grad4.z();

    for(int i=0; i< 4; i++)
      for(int j=0; j< 4; j++) {
	for (int k=0; k< 3; k++)
	  lcl_a[i][j] += el_coefs[i][k]*el_coefs[j][k];
	lcl_a[i][j] *= vol;
      }

    // now add these values into dataC[c]
    c=elem->cond;

    for (i=0; i<4; i++) {	  
      int ii = elem->n[i];
      if (ii!=refnode || !pinzero)
	for (int j=0; j<4; j++) {
	  int jj = elem->n[j];
	  if (jj!=refnode || !pinzero) {
	    int idx=AmatHp->getIdx(ii,jj);
	    dataC[c][idx] += lcl_a[i][j];
	  }
	}
      else {
	int idx=AmatHp->getIdx(ii,ii);
	dataC[c][idx] = 1;
      }
    }
  }

  // go through each conductivity type and add nonzero dataC indices to nzeros
  for (c=0; c<ncond; c++) {
    for (i=0; i<AmatHp->nnz; i++)
      if (dataC[c][i] != 0) 
	nzeros[c].add(i);
  }
}

void ConductivitySearch::execute() {
  if (state != "START") {
    cerr << "Sending last result again...\n";
    mat_oport->send(AmatH);
    cond_oport->send(conds);
    return;
  }

  if(!mesh_iport->get(mesh) || !mesh.get_rep()) {
    cerr << "ConductivitySearch -- couldn't get mesh.  Returning.\n";
    return;
  }

  MatrixHandle AH;
  if (!mat_iport->get(AH) || !AH.get_rep()) {
    cerr << "ConductivitySearch -- couldn't get stiffness matrix.\n";
    return;
  }

  _CS_nCondReg = mesh->cond_tensors.size();

  cerr << "Getting refnode and pinzero...\n";
  refnodeTCL.get().get_int(refnode);
  pinzero = pinzeroTCL.get();
  cerr << "pinzero="<<pinzero<<"  refnode="<<refnode<<"\n";
  int i, j;
  int newSeed;
  seedTCL.get().get_int(newSeed);
  if (AH->generation != AHgen || seed != newSeed) {
    SparseRowMatrix *AHp=dynamic_cast<SparseRowMatrix*>(AH.get_rep());
    if (!AHp) {
      cerr << "Error - A matrix wasn't a SparseRowMatrix!\n";
      return;
    }
    seed=newSeed;
    seedTCL.set(to_string(seed+1));
    mr = new MusilRNG(seed);
    (*mr)();        // first number isn't random
    AHgen=AH->generation;
    // because the SparseRowMatrix copy constructor doesn't exist yet...
    Array1<int> rows(AHp->nrows()+1);
    Array1<int> cols(AHp->nnz);
    for (i=0; i<rows.size(); i++) rows[i]=AHp->rows[i];
    for (i=0; i<cols.size(); i++) cols[i]=AHp->columns[i];    
    AmatH=AmatHp=new SparseRowMatrix(AHp->nrows(),AHp->ncols(),rows,cols);
    buildCondMatrices(_CS_nCondReg);
  }

  conds=new ColumnMatrix(_CS_nCondReg*2);
  ColumnMatrix *condsp = conds.get_rep();

  for (i=0; i<_CS_nCondReg; i++) (*condsp)[i]=mesh->cond_tensors[i][0];

  if(!cond_iport->get(_CS_cond_params)) {
    cerr << "ConductivitySearch -- couldn't get conductivity parameters.  Returning.\n";
    return; 
  }
  if (_CS_cond_params->nrows() != _CS_nCondReg) {
    cerr << "Error - need same number of mesh conductivity regions ("<<_CS_nCondReg<<") as conductivity parameter rows ("<<_CS_cond_params->nrows()<<".\n";
    return;
  }
  if (_CS_cond_params->ncols() != 4) {
    cerr <<"Error - found "<<_CS_cond_params->ncols()<<" parameters per conductivity region, but we need 4: mean, sigma, min, max.\n";
    return;
  }

  _CS_p = dmatrix(1, _CS_nCondReg+2, 1, _CS_nCondReg);
  y = dvector(1, _CS_nCondReg);
    
  cerr << "Choosing starting conductivities...\n";
  // load Gaussian distribution of conductivity values into p
  for (i=0; i<_CS_nCondReg; i++) { // cond regions
    for (j=0; j<_CS_nCondReg+2; j++) { // amoeba vertex (last won't be used)
      double val, min, max;
      do 
      {
	double g, sigma, avg;
	avg=(*(_CS_cond_params.get_rep()))[i][0];
	sigma=(*(_CS_cond_params.get_rep()))[i][1];
	g=gaussian(sigma);
	min=(*(_CS_cond_params.get_rep()))[i][2];
	max=(*(_CS_cond_params.get_rep()))[i][3];
	val= _CS_p[j+1][i+1] = avg + g;
//	cerr << "i="<<i<<" j="<<j<<"  avg="<<  avg<<"  sigma="<<sigma<<"  g="<<g<<"  min="<<min<<"  max="<<max<<"  val="<<val<<"\n";
      } while (val > max || val < min);
    }
  }
  cerr << "Starting conductivities...\n";
  for (i=0; i<_CS_nCondReg+1; i++) {
    cerr << "node "<<i+1 << ":  ";
    for (j=0; j<_CS_nCondReg; j++)
      cerr << _CS_p[i+1][j+1] << " ";
    cerr << "\n";
  }

  _CS_send_pos = 1;
  _CS_need_error = 1;
  _CS_in_bounds = 1;

  int num_evals=1;
  int count=1;

  Array1<double> curr_conds(_CS_nCondReg);
  int ii;
  for (ii=0; ii<_CS_nCondReg; ii++)
    (*condsp)[ii+_CS_nCondReg]=curr_conds[ii]=_CS_p[1][ii+1];
  buildCompositeMat(curr_conds);

  mat_oport->send_intermediate(AmatH);
  cond_oport->send_intermediate(conds);

  cerr << "conductivities = ";
  for (ii=0; ii<_CS_nCondReg; ii++)
    cerr << curr_conds[ii]<<" ";
  cerr << "\n";

  counter=1;
  while(1) {
    count++;
    //	cerr << "ABOUT TO TRY LOCK\n";
    if (!mylock.tryLock()) {
      //	    cerr << "ConductivitySearch -- Pausing...\n";
      mylock.lock();
      mylock.unlock();
      //	    cerr << "ConductivitySearch -- Unpausing...\n";
    } else {
      //	    cerr << "ConductivitySearch -- got the lock, but I don't want it...";
      mylock.unlock();
      //	    cerr << " so I just unlocked it.\n";
    }
    //	cerr << "DONE TRYING LOCK\n";
	
    //	cerr <<"Getting new error!"<<endl;
    ColumnMatrixHandle rms;

    if (_CS_need_error) {
      num_evals++;
      if(!rms_iport->get(rms)) {
	cerr << "Error - ConductivitySearch didn't get an errorMetric!\n";
	return;
      }
      //	cerr << "ConductivitySearch: got errormetric.\n";
    } else {
      rms = new ColumnMatrix(1);
      double *rms_err = rms->get_rhs();
      cerr << "ConductivitySearch - using out of bounds error.\n";
      rms_err[0] = 1000000;
    }

    //	cerr << "ConductivitySearch -- got the data!!\n";
    double *rms_err = rms->get_rhs();
	
    _CS_err=new double[1];
    _CS_err[0]=rms_err[0];

    cerr << "ConductivitySearch: error=" << _CS_err[0] << "\n";
	
    if (counter>(_CS_nCondReg+1)) {
      _CS_error_sem->up();
    }

    if (counter <= _CS_nCondReg) {
      for (ii=0; ii<_CS_nCondReg; ii++)
	(*condsp)[ii+_CS_nCondReg]=curr_conds[ii]=_CS_p[counter+1][ii+1];
      buildCompositeMat(curr_conds);
      y[counter]=_CS_err[0];
      counter++;
    } else {
      if (counter == (_CS_nCondReg+1)) {
	y[counter]=_CS_err[0];
	for (ii=0; ii<_CS_nCondReg; ii++)
	  (*condsp)[ii+_CS_nCondReg]=curr_conds[ii]=_CS_p[counter+1][ii+1];
	buildCompositeMat(curr_conds);
	_CS_helper_sem->up();
	counter++;
      }
      _CS_pos_sem->down();
    }	
    if (counter>(_CS_nCondReg+1)) {
      for (ii=0; ii<_CS_nCondReg; ii++)
	(*condsp)[ii+_CS_nCondReg]=curr_conds[ii]=_CS_p[_CS_nCondReg+2][ii+1];
      buildCompositeMat(curr_conds);
    }

    if (_CS_need_error) {
      if (_CS_send_pos) {
	mat_oport->send_intermediate(AmatH);
	cond_oport->send_intermediate(conds);
      } else {
	cerr << "I'm sending final version\n";
	mat_oport->send(AmatH);
	cond_oport->send(conds);
		
	// gotta clear this port so when we start again everything works
	rms_iport->get(rms);
	break;
      }
    }
  }
  cerr << "Done downhill!   num_evals="<<num_evals<<"  count="<<count<<"\n";
  state = "DONE";
}


// helper is a second thread that runs the Numerical Recipes amoeba code
//   the main thread controls marshalling for port I/O, and startup/shoutdown 
// the two threads communicate through semaphores - _CS_pos_sem and _CS_error_sem
// the main thread waits on _CS_pos_sem - the amoeba has to tell it what position
//   it wants evaluated
// the helper thread waits on _CS_error_sem - it wants to know the error for
//   a new position, which the main thread evaluates by sending/receiving port
//   data
//---------------------------------------------------------------
void ConductivitySearch::helper(int /*proc*/){
  while(1){
    _CS_helper_sem->down();
	
    cerr <<"Calling amoeba()"<<endl;
    FTOL = 1.0e-10;
    nfunc=200;
	
    int i;
    for (i=1; i<=(_CS_nCondReg+2); i++) {
      printf("%3d ",i);
      for (int j=1;j<=_CS_nCondReg;j++) printf("%12.6f ",_CS_p[i][j]);
      printf("%12.6f\n",y[i]);
    }
	
    cerr <<"_CS_nCondReg = "<<_CS_nCondReg<<endl;
	
    _CS_stop=0;
    amoeba(_CS_p, y, _CS_nCondReg, FTOL, ConductivitySearch::error_eval,
	   &nfunc,0,&_CS_stop);
	
    printf("\nNumber of function evaluations: %3d\n",nfunc);
    printf("function values at the vertices:\n\n");
    for (i=1; i<=(_CS_nCondReg+2); i++) {
      printf("%3d ",i);
      for (int j=1;j<=_CS_nCondReg;j++) printf("%12.6f ",_CS_p[i][j]);
      printf("%12.6f\n",y[i]);
    }
	
    _CS_send_pos=0;
    _CS_in_bounds = 1;
    _CS_need_error = 1;
    _CS_pos_sem->up();
  }
}

// this is the method that the amoeba calls when it needs to know the
//   error for a particular configuration.
// the "hidden" variables that are being used are "p", which is a 
//   matrix containing all of the conductivity configurations; and  
//   "y" contains the error for each configuration in p
//---------------------------------------------------------------
double* ConductivitySearch::error_eval(int) {
  _CS_send_pos = 1;
  _CS_in_bounds = 1;
  _CS_need_error = 1;

  int i,j;

  // check to see if the new conductivities are within the min-max ranges

  for (i=0; i<_CS_nCondReg; i++) 
    if (_CS_p[_CS_nCondReg+2][i+1] < (*(_CS_cond_params.get_rep()))[i][2] || 
	_CS_p[_CS_nCondReg+2][i+1] > (*(_CS_cond_params.get_rep()))[i][3]) {
      cerr << "Cond["<<i<<"] = "<<_CS_p[_CS_nCondReg+2][i+1]<<"\n";
      _CS_in_bounds=0;
    }

  // check to see if we have converged
  double resid=0;
  for (i=0; i<_CS_nCondReg; i++)
    for (j=1; j<_CS_nCondReg+1; j++)
      resid += fabs(_CS_p[1][i+1]-_CS_p[j+1][i+1]);
  if (resid/(_CS_nCondReg*(_CS_nCondReg+1)) < 0.00005) {
    cerr << "Resid="<<resid<<" -- we've converged!\n";
    _CS_stop=1;
  }

  if (_CS_in_bounds) {
    _CS_pos_sem->up();
    _CS_error_sem->down();
  } else {
    cerr << "Conductivity out of bounds.\n";
    _CS_need_error = 0;
    _CS_pos_sem->up();
    _CS_error_sem->down();
  }
  return _CS_err;
}

void ConductivitySearch::tcl_command(TCLArgs& args, void* userdata) {
  if (args[1] == "pause") {
    if (mylock.tryLock())
      cerr << "Pausing...\n";
    else 
      cerr << "Can't lock -- already locked!\n";
  } else if (args[1] == "unpause") {
    if (mylock.tryLock())
      cerr << "Can't unlock -- already unlocked!\n";
    else
      cerr << "Unpausing.\n";
    mylock.unlock();
  } else if (args[1] == "stop") {
    _CS_stop=1;
  } else if (args[1] == "print") {
    int i,j;
    for (j=0; j<_CS_nCondReg+1; j++) { // amoeba vertex (last won't be used)
      cerr << j+1 << "\t";
      for (i=0; i<_CS_nCondReg; i++)   // cond regions
	cerr << _CS_p[j+1][i+1] << "\t";
      cerr << "error="<<y[j+1]<<"\n";
    }
  } else if (args[1] == "exec") {
    state = "START";
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace DaveW
//---------------------------------------------------------------


