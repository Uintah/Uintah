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

/****************************************************************
 *  Simple "Downhill_Simplex3 module" for Dataflow                *
 *                                                              *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   May 1999                                                   *
 *                                                              *
 *  Copyright (C) 1999 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#include <Packages/DaveW/ThirdParty/NumRec/amoeba.h>
#include <Packages/DaveW/ThirdParty/NumRec/protozoa.h>
#include <Packages/DaveW/ThirdParty/NumRec/nrutil.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/SymSparseRowMatrix.h>
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
    
double ** p; //holds initial dipole configuration 
int ndipoles; 
Array1<int> cell_visited;
Array1<double> cell_err;
Array1<Vector> cell_dir;
int useCache;
int curr_idx;
int in_bounds;
int need_error;
int ndim;
int send_pos;
int amoeba_stop;
int use_protozoa;
int posIdx;
double *err;
Semaphore *error_sem, *pos_sem, *helper_sem;
MeshHandle mesh_in;
MusilRNG mr;    
//int ndim;
//int send_pos; 
//double* err;
//Semaphore *error_sem, *pos_sem, *helper_sem;

class Downhill_Simplex3 : public Module {    
    MatrixIPort* dipole_inport;
    ColumnMatrixIPort* rms_inport;
    ColumnMatrixIPort* dir_inport;
    MeshIPort* mesh_inport;
    MatrixOPort* dip_eval_oport;
    MatrixOPort* dip_vis_oport;
    ColumnMatrixOPort* best_errors_oport;
    double FTOL;
    int nfunc;
    int counter;
    clString state;
    int first;
    double* y; //holds initial errors for p configuration
public:
    Mutex mylock;
    GuiString tcl_status;
    GuiString methodTCL;
    GuiInt useCacheTCL;
    Downhill_Simplex3(const clString& id);
    virtual ~Downhill_Simplex3();
    virtual void execute();
    Vector gaussian(double w);
    int metro(double dE, double T);
    void executeDownhill();
    void executeProtozoa();
    void executeAnneal();
    void executeRoll();
    static double *error_eval(int);
    void helper(int proc);
    void tcl_command( TCLArgs&, void * );
}; //class


extern "C" Module* make_Downhill_Simplex3(const clString& id) {
    return new Downhill_Simplex3(id);
}

//---------------------------------------------------------------
Downhill_Simplex3::Downhill_Simplex3(const clString& id)
: Module("Downhill_Simplex3", id, Filter), tcl_status("tcl_status",id,this),
  mylock("pause lock for Downhill_Simplex3"), methodTCL("methodTCL",id,this),
  useCacheTCL("useCacheTCL",id,this)
{
    rms_inport = new ColumnMatrixIPort(this, "RMS port",
				       ColumnMatrixIPort::Atomic);
    add_iport(rms_inport);
    
    dipole_inport = new MatrixIPort(this, "Dipole In port",
				    MatrixIPort::Atomic);
    add_iport(dipole_inport);
    
    dir_inport = new ColumnMatrixIPort(this,"Direction port",
				       ColumnMatrixIPort::Atomic);
    add_iport(dir_inport);
    
    mesh_inport = new MeshIPort(this,"Mesh In port",
				MeshIPort::Atomic);
    add_iport(mesh_inport);
    
    dip_eval_oport = new MatrixOPort(this, "DipolePos Out port",
				MatrixIPort::Atomic);
    add_oport(dip_eval_oport);
//    dipoley_outport = new ColumnMatrixOPort(this, "DipoleY Out port",
//					    ColumnMatrixIPort::Atomic);
//    add_oport(dipoley_outport);
//    dipolez_outport = new ColumnMatrixOPort(this, "DipoleZ Out port",
//					    ColumnMatrixIPort::Atomic);
//    add_oport(dipolez_outport);

    dip_vis_oport = new MatrixOPort(this, "DipoleMat Out port",
				    MatrixIPort::Atomic);
    add_oport(dip_vis_oport);
    
    best_errors_oport = new ColumnMatrixOPort(this, "Best Errors", 
					      ColumnMatrixIPort::Atomic);
    add_oport(best_errors_oport);

    counter = 0;
    state = "START";
    ndim = 3;
    ndipoles = (ndim + 1);
    
    error_sem = new Semaphore("Downhill_Simplex3 error sync", 0);
    pos_sem = new Semaphore("Downhill_Simplex3 position sync", 0);
    helper_sem = new Semaphore("Downhill_Simplex3 amoeba sync", 0);
    
    Thread::parallel(Parallel<Downhill_Simplex3>(this, &Downhill_Simplex3::helper), 1, false);
//    Task::multiprocess(1, start_me_up, this, false);
    
    cerr<<"Constructor Done!"<<endl;
    mylock.unlock();
    first=1;
    curr_idx=0;
}

//------------------------------------------------------------
Downhill_Simplex3::~Downhill_Simplex3(){}

//--------------------------------------------------------------

void Downhill_Simplex3::executeDownhill() {
    MatrixHandle dipole_in;
    if(!dipole_inport->get(dipole_in)) {
	cerr << "Downhill_Simplex3 -- couldn't get dipoles.  Returning.\n";
	return; 
    }
    if (dipole_in->nrows() < ndipoles) {
	cerr << "Error - need at least "<<ndipoles<<" dipoles as input!\n";
	return;
    }
    if (dipole_in->ncols() < 3) {
	cerr <<"Error - need at least 3 coords (x,y,z) for each dipole!\n";
	return;
    }

    p = dmatrix(1,ndipoles+1,1,ndim+3);
    y = dvector(1,ndim);
    
    send_pos = 1;
    need_error = 1;
    in_bounds = 1;

    int num_evals=1;
    int count=1;
    cell_visited.resize(mesh_in->elems.size());
    cell_err.resize(mesh_in->elems.size());
    cell_dir.resize(mesh_in->elems.size());
    cell_visited.initialize(0);

    int i;
    for (i=0; i<ndipoles; i++) {
	for (int j=0; j<3; j++) {
	    p[i+1][j+1]=(*dipole_in.get_rep())[i][j];
	}
	p[i+1][4]=p[i+1][5]=0;
	p[i+1][6]=1;
    }
    p[i+1][1]=p[i+1][2]=p[i+1][3]=p[i+1][4]=p[i+1][5]=p[i+1][6]=0;
    
    DenseMatrix* dip_eval_out = new DenseMatrix(6,3);
    dip_eval_out->zero();
    for (int ii=0; ii<3; ii++) {
	for (int jj=0; jj<3; jj++)
	    (*dip_eval_out)[ii][jj]=p[1][ii+1];
	(*dip_eval_out)[ii+3][ii]=1;
    }

    cerr << "Downhill_Simplex3: about to do a send_intermediate...\n";
    dip_eval_oport->send_intermediate(dip_eval_out);
    cerr << "Downhill_Simplex3: sent intermediate!\n";
    
    cerr << "dip_eval_out=("<<(*dip_eval_out)[0][0]<<", "<<(*dip_eval_out)[1][0]<<", "<<(*dip_eval_out)[2][0]<<")\n";
    
    counter=1;
    while(1) {
	count++;
//	cerr << "ABOUT TO TRY LOCK\n";
	if (!mylock.tryLock()) {
//	    cerr << "Downhill_Simplex3 -- Pausing...\n";
	    mylock.lock();
	    mylock.unlock();
//	    cerr << "Downhill_Simplex3 -- Unpausing...\n";
	} else {
//	    cerr << "Downhill_Simplex3 -- got the lock, but I don't want it...";
	    mylock.unlock();
//	    cerr << " so I just unlocked it.\n";
	}
//	cerr << "DONE TRYING LOCK\n";
	
//	cerr <<"Getting new error!"<<endl;
	ColumnMatrixHandle rms;
	ColumnMatrixHandle dir;
	MatrixHandle dipole_in;

	if (need_error) {
	    num_evals++;
	    cerr << "DS2 - getting error from port.\n";
	    if(!dir_inport->get(dir)) {
		cerr << "Error - Downhill_Simplex3 didn't get a direction!\n";
		return;
	    }
	    //	cerr << "Downhill_Simplex3: got direction.\n";
	    if(!rms_inport->get(rms)) {
		cerr << "Error - Downhill_Simplex3 didn't get an errorMetric!\n";
		return;
	    }
	    //	cerr << "Downhill_Simplex3: got errormetric.\n";
	} else {
	    rms = new ColumnMatrix(1);
	    double *rms_err = rms->get_rhs();
	    dir = new ColumnMatrix(3);
	    double *dirP = dir->get_rhs();
	    if (in_bounds) {
		cerr << "DS2 - using lookup error.\n";
		rms_err[0] = cell_err[curr_idx];
		dirP[0] = cell_dir[curr_idx].x();
		dirP[1] = cell_dir[curr_idx].y();
		dirP[2] = cell_dir[curr_idx].z();
	    } else {
		cerr << "DS2 - using out of bounds error.\n";
		rms_err[0] = 1000000;
		dirP[0] = 1; dirP[1] = 0; dirP[2] = 0;
	    }
	}

//	cerr << "Downhill_Simplex3 -- got the data!!\n";
	double *rms_err = rms->get_rhs();
	double *dirP=dir->get_rhs();
	
	err=new double[4];
	err[0]=rms_err[0]; 
	err[1]=dirP[0]; err[2]=dirP[1]; err[3]=dirP[2];
	cerr << "Downhill_Simplex3: Direction=(" << err[1] << "," << err[2] << "," << err[3];
	cerr << " ) - error=" << err[0] << "\n";
	
	if (counter>ndipoles) {
	    error_sem->up();
	}
	
	dip_eval_out = new DenseMatrix(7, 3);
	dip_eval_out->zero();
	

	int ii=0;
	if (useCache && cell_visited[curr_idx]) 
	    for (; ii<3; ii++) (*dip_eval_out)[6][ii]=curr_idx;
	ii=0;
	if (counter < ndipoles) {
	    y[counter]=*err;
	    for (; ii<3; ii++) {
		for (int jj=0; jj<3; jj++) 
		    (*dip_eval_out)[ii][jj]=p[counter+1][ii+1];
		(*dip_eval_out)[ii+3][ii]=1;
	    }
	    for (; ii<6; ii++) p[counter][ii+1]=dirP[ii-3];
	    counter++;
	} else {
	    for (ii=3; ii<6; ii++) p[counter][ii+1]=dirP[ii-3];
	    if (counter == ndipoles) {
		y[counter]=*err;
		for (ii=0; ii<3; ii++) {
		    for (int jj=0; jj<3; jj++) 
			(*dip_eval_out)[ii][jj]=p[counter+1][ii+1];
		    (*dip_eval_out)[ii+3][ii]=1;		    
		}
		use_protozoa=0;
		helper_sem->up();
		counter++;
	    }
	    pos_sem->down();
	}	
	
	if (counter>ndipoles) 
	    for (ii=0; ii<3; ii++) {
		for (int jj=0; jj<3; jj++)
		    (*dip_eval_out)[ii][jj]=p[posIdx][ii+1];
		(*dip_eval_out)[ii+3][ii]=1;
	    }

	DenseMatrix* dip_vis_out = new DenseMatrix(ndipoles+1, ndim+3);
	MatrixHandle dip_vis_outH(dip_vis_out);

//	cerr << "Here are the dipoles (straight from p):\n";
	for (ii=0; ii<ndipoles+1; ii++) {
//	    cerr << "   "<<ii+1<<" = ";
	    for (int jj=0; jj<ndim+3; jj++) {
		(*dip_vis_out)[ii][jj]=p[ii+1][jj+1];
//		cerr << p[ii+1][jj+1] << " ";
	    }
//	    cerr << "\n";
	}


//	if (need_error) {
//	    cerr << "DS2 - sending position for vis...\n";
	    if (send_pos)
		dip_vis_oport->send_intermediate(dip_vis_outH);
	    else
		dip_vis_oport->send(dip_vis_outH);
//	} else {
//	    cerr << "DS2 - not sending position...\n";
//	}

	if (need_error) {
	if (send_pos) {
	    dip_eval_oport->send_intermediate(dip_eval_out);
	} else {
	    cerr << "I'm sending final version\n";
	    dip_eval_oport->send(dip_eval_out);

	    // gotta clear these ports so when we start again, everything works
	    dir_inport->get(dir);
	    rms_inport->get(rms);
	    break;
	}
	}
    }
    cerr << "Done downhill!   num_evals="<<num_evals<<"  count="<<count<<"\n";
}

void Downhill_Simplex3::executeProtozoa() {
    MatrixHandle dipole_in;
    if(!dipole_inport->get(dipole_in)) {
	cerr << "Downhill_Simplex3 -- couldn't get dipoles.  Returning.\n";
	return; 
    }
    if (dipole_in->nrows() < ndim*2+1) {
	cerr << "Error - need at least "<<ndim*2+1<<" dipoles as input!\n";
	return;
    }
    if (dipole_in->ncols() < 3) {
	cerr <<"Error - need at least 3 coords (x,y,z) for each dipole!\n";
	return;
    }

    p = dmatrix(1,2*ndim+1,1,ndim+3);
    y = dvector(1,2*ndim+1);
    
    send_pos = 1;
    need_error = 1;
    in_bounds = 1;

    int num_evals=1;
    int count=1;
    cell_visited.resize(mesh_in->elems.size());
    cell_err.resize(mesh_in->elems.size());
    cell_dir.resize(mesh_in->elems.size());
    cell_visited.initialize(0);

    int i;
    for (i=0; i<2*ndim+1; i++) {
	for (int j=0; j<3; j++) {
	    p[i+1][j+1]=(*dipole_in.get_rep())[i][j];
	}
	p[i+1][4]=p[i+1][5]=0;
	p[i+1][6]=1;
    }
    
    DenseMatrix* dip_eval_out = new DenseMatrix(6,3);
    dip_eval_out->zero();
    for (int ii=0; ii<3; ii++) {
	for (int jj=0; jj<3; jj++)
	    (*dip_eval_out)[ii][jj]=p[1][ii+1];
	(*dip_eval_out)[ii+3][ii]=1;
    }

    cerr << "Downhill_Simplex3: about to do a send_intermediate...\n";
    dip_eval_oport->send_intermediate(dip_eval_out);
    cerr << "Downhill_Simplex3: sent intermediate!\n";
    
    cerr << "dip_eval_out=("<<(*dip_eval_out)[0][0]<<", "<<(*dip_eval_out)[1][0]<<", "<<(*dip_eval_out)[2][0]<<")\n";
    
    counter=1;
    int first=1;
    while(1) {
	count++;
//	cerr << "ABOUT TO TRY LOCK\n";
	if (!mylock.tryLock()) {
//	    cerr << "Downhill_Simplex3 -- Pausing...\n";
	    mylock.lock();
	    mylock.unlock();
//	    cerr << "Downhill_Simplex3 -- Unpausing...\n";
	} else {
//	    cerr << "Downhill_Simplex3 -- got the lock, but I don't want it...";
	    mylock.unlock();
//	    cerr << " so I just unlocked it.\n";
	}
//	cerr << "DONE TRYING LOCK\n";
	
//	cerr <<"Getting new error!"<<endl;
	ColumnMatrixHandle rms;
	ColumnMatrixHandle dir;
	MatrixHandle dipole_in;

	if (need_error) {
	    num_evals++;
	    cerr << "DS2 - getting error from port.\n";
	    if(!dir_inport->get(dir)) {
		cerr << "Error - Downhill_Simplex3 didn't get a direction!\n";
		return;
	    }
	    //	cerr << "Downhill_Simplex3: got direction.\n";
	    if(!rms_inport->get(rms)) {
		cerr << "Error - Downhill_Simplex3 didn't get an errorMetric!\n";
		return;
	    }
	    //	cerr << "Downhill_Simplex3: got errormetric.\n";
	} else {
	    rms = new ColumnMatrix(1);
	    double *rms_err = rms->get_rhs();
	    dir = new ColumnMatrix(3);
	    double *dirP = dir->get_rhs();
	    if (in_bounds) {
		cerr << "DS2 - using lookup error.\n";
		rms_err[0] = cell_err[curr_idx];
		dirP[0] = cell_dir[curr_idx].x();
		dirP[1] = cell_dir[curr_idx].y();
		dirP[2] = cell_dir[curr_idx].z();
	    } else {
		cerr << "DS2 - using out of bounds error.\n";
		rms_err[0] = 1000000;
		dirP[0] = 1; dirP[1] = 0; dirP[2] = 0;
	    }
	}

//	cerr << "Downhill_Simplex3 -- got the data!!\n";
	double *rms_err = rms->get_rhs();
	double *dirP=dir->get_rhs();
	
	err=new double[4];
	err[0]=rms_err[0]; 
	err[1]=dirP[0]; err[2]=dirP[1]; err[3]=dirP[2];
	cerr << "Downhill_Simplex3: Direction=(" << err[1] << "," << err[2] << "," << err[3];
	cerr << " ) - error=" << err[0] << "\n";
	
	if (counter==ndim*2+1 && !first) {
	    error_sem->up();
	}
	
	dip_eval_out = new DenseMatrix(7, 3);
	dip_eval_out->zero();
	
	int ii=0;
	if (useCache && cell_visited[curr_idx]) 
	    for (; ii<3; ii++) (*dip_eval_out)[6][ii]=curr_idx;
	ii=0;
	if (counter < ndim*2+1) {
	    y[counter]=*err;
	    for (; ii<3; ii++) {
		for (int jj=0; jj<3; jj++) 
		    (*dip_eval_out)[ii][jj]=p[counter+1][ii+1];
		(*dip_eval_out)[ii+3][ii]=1;
	    }
	    for (; ii<6; ii++) p[counter][ii+1]=dirP[ii-3];
	    counter++;
	} else {
	    for (ii=3; ii<6; ii++) p[counter][ii+1]=dirP[ii-3];
	    if (counter == ndim*2+1 && first) {
		y[counter]=*err;
		for (ii=0; ii<3; ii++) {
		    for (int jj=0; jj<3; jj++) 
			(*dip_eval_out)[ii][jj]=p[counter][ii+1];
		    (*dip_eval_out)[ii+3][ii]=1;		    
		}
		first=0;
		use_protozoa=1;
		helper_sem->up();
//		counter++;
	    }
	    pos_sem->down();
	}	
	
	if (counter==ndim*2+1 && !first)
	    for (ii=0; ii<3; ii++) {
		for (int jj=0; jj<3; jj++)
		    (*dip_eval_out)[ii][jj]=p[posIdx][ii+1];
		(*dip_eval_out)[ii+3][ii]=1;
	    }

	DenseMatrix* dip_vis_out = new DenseMatrix(ndim*2+1, ndim+3);
	MatrixHandle dip_vis_outH(dip_vis_out);

//	cerr << "Here are the dipoles (straight from p):\n";
	for (ii=0; ii<ndim*2+1; ii++) {
//	    cerr << "   "<<ii+1<<" = ";
	    for (int jj=0; jj<ndim+3; jj++) {
		(*dip_vis_out)[ii][jj]=p[ii+1][jj+1];
//		cerr << p[ii+1][jj+1] << " ";
	    }
//	    cerr << "\n";
	}


//	if (need_error) {
//	    cerr << "DS2 - sending position for vis...\n";
	    if (send_pos)
		dip_vis_oport->send_intermediate(dip_vis_outH);
	    else
		dip_vis_oport->send(dip_vis_outH);
//	} else {
//	    cerr << "DS2 - not sending position...\n";
//	}

	if (need_error) {
	if (send_pos) {
	    dip_eval_oport->send_intermediate(dip_eval_out);
	} else {
	    cerr << "I'm sending final version\n";
	    dip_eval_oport->send(dip_eval_out);

	    // gotta clear these ports so when we start again, everything works
	    dir_inport->get(dir);
	    rms_inport->get(rms);
	    break;
	}
	}
    }
    cerr << "Done downhill!   num_evals="<<num_evals<<"  count="<<count<<"\n";
}

Vector Downhill_Simplex3::gaussian(double sigma) {
    double x1, x2, x3, w;
    do {
	x1 = 2.0 * mr() - 1.0;
	x2 = 2.0 * mr() - 1.0;
	x3 = 2.0 * mr() - 1.0;
	w = x1 * x1 + x2 * x2 + x3 * x3;
    } while ( w >= 1.0 );
    w = sqrt( (-2.0 * log( w ) ) / w );
    return Vector(x1*w*sigma, x2*w*sigma, x3*w*sigma);
}

int Downhill_Simplex3::metro(double dE, double T) {
    if (dE < 0) return 1;
    if (mr() < exp(-dE/T)) {
	cerr << "\n\nMETRO: exp(-dE/T) = exp("<<-dE<<"/"<<T<<") = "<<exp(-dE/T)<<"\n\n\n";
	return 1;
    }
    return 0;
}

void Downhill_Simplex3::executeAnneal() {
    Array1<double> best_errors;
    cell_visited.resize(mesh_in->elems.size());
    cell_err.resize(mesh_in->elems.size());
    cell_dir.resize(mesh_in->elems.size());
    cell_visited.initialize(0);
    //cell_err;
    //cell_dir;
    //curr_idx;
    
    amoeba_stop=0;
    Point min, max, best_p;
    int best_idx;
    mesh_in->get_bounds(min, max);
    Point p(min+(max-min)/2);
    
    Vector diag((max-min)/pow(mesh_in->elems.size(), .3));
//    double gWidth = (max-min).length()/2;
    double best_err, T=1;
    int have_first_err=0;
    int count=0;
    ColumnMatrixHandle rms;
    ColumnMatrixHandle dir;

    int num_evals=0;

    int cycle_idx=0;
    DenseMatrix dips(5,6);
    for (int i=0; i<5; i++) {
	dips[i][0]=p.x(); dips[i][1]=p.y();
	dips[i][2]=p.z(); dips[i][3]=dips[i][4]=0; dips[i][5]=0.001;
    }

    // while the error is above threshold, find a new dipole position, 
    //       and evaluate its error
    do {
	double err1;
	Vector optDir;
	if (!mylock.tryLock()) {
	    mylock.lock();
	    mylock.unlock();
	} else {
	    mylock.unlock();
	}
	
	Vector v((mr()*2-1)*diag.x(),(mr()*2-1)*diag.y(),(mr()*2-1)*diag.z());
	p = p+v;
	cerr << "p="<<p<<" ";
	if (mesh_in->locate(p, curr_idx)) {
	    cerr << "Found p in e="<<curr_idx<<" ";
	    if (!useCache || !cell_visited[curr_idx]) {
		cell_visited[curr_idx]=1;
		DenseMatrix *dip_eval_out = new DenseMatrix(7,3);
		dip_eval_out->zero();
		int ii=0;
		if (useCache && cell_visited[curr_idx]) 
		    for (; ii<3; ii++) (*dip_eval_out)[6][ii]=curr_idx;
		(*dip_eval_out)[0][0]=(*dip_eval_out)[0][1]=(*dip_eval_out)[0][2]=p.x();
		(*dip_eval_out)[1][0]=(*dip_eval_out)[1][1]=(*dip_eval_out)[1][2]=p.y();
		(*dip_eval_out)[2][0]=(*dip_eval_out)[2][1]=(*dip_eval_out)[2][2]=p.z();
		(*dip_eval_out)[3][0]=(*dip_eval_out)[4][1]=(*dip_eval_out)[5][2]=1;
		dip_eval_oport->send_intermediate(dip_eval_out);

		// vis junk
		dips[4][0]=p.x();
		dips[4][1]=p.y();
		dips[4][2]=p.z();
		dips[4][3]=(dips[0][3]+dips[1][3]+dips[2][3]+dips[3][3])/3.;
		dips[4][4]=(dips[0][4]+dips[1][4]+dips[2][4]+dips[3][4])/3.;
		dips[4][5]=(dips[0][5]+dips[1][5]+dips[2][5]+dips[3][5])/3.;
		DenseMatrix *vis_out = new DenseMatrix(5,6);
		for (ii=0; ii<5; ii++) 
		    for (int jj=0; jj<6; jj++) 
			(*vis_out)[ii][jj]=dips[ii][jj];
		MatrixHandle voH(vis_out);
		dip_vis_oport->send_intermediate(MatrixHandle(voH));

		if(!dir_inport->get(dir)) {
		    cerr << "Error - Downhill_Simplex3 didn't get a dir!\n";
		    return;
		}
		if(!rms_inport->get(rms)) {
		    cerr << "Error - Downhill_Simplex3 didn't get a metric!\n";
		    return;
		}
		double *ee=rms->get_rhs();
		if (rms->nrows() != 1)
		    cerr << "Error - rms->nrows="<<rms->nrows()<<"\n";
		cell_err[curr_idx]=err1=ee[0];
		cell_dir[curr_idx]=Vector(dir->get_rhs()[0], dir->get_rhs()[1],
					  dir->get_rhs()[2]);
		num_evals++;
	    } else {
		dips[4][0]=p.x();
		dips[4][1]=p.y();
		dips[4][2]=p.z();
		dips[4][3]=cell_dir[curr_idx].x();
		dips[4][4]=cell_dir[curr_idx].y();
		dips[4][5]=cell_dir[curr_idx].z();
		DenseMatrix *vis_out = new DenseMatrix(5,6);
		for (int ii=0; ii<5; ii++) 
		    for (int jj=0; jj<6; jj++) 
			(*vis_out)[ii][jj]=dips[ii][jj];
		MatrixHandle voH(vis_out);
		dip_vis_oport->send_intermediate(MatrixHandle(voH));

		cerr << "Found in element: "<<curr_idx<<" REUSING CACHED ERROR.\n";
		err1=cell_err[curr_idx];
	    }
	    cerr << "e="<<err1<<" ";
	    if (!have_first_err || metro(err1-best_err, T)) {
		dips[cycle_idx][0]=p.x();
		dips[cycle_idx][1]=p.y();
		dips[cycle_idx][2]=p.z();
		dips[cycle_idx][3]=cell_dir[curr_idx].x();
		dips[cycle_idx][4]=cell_dir[curr_idx].y();
		dips[cycle_idx][5]=cell_dir[curr_idx].z();
		cycle_idx=(cycle_idx+1) % 4;
		best_err=err1;
		best_p=p;
		best_idx=curr_idx;
		have_first_err=1;
	    } else p=p-v;
	    cerr << "BestE="<<best_err<<"  BestP="<<best_p<<" ";
//	    gWidth=gWidth*.9;
	    if (count && (count%10 == 0)) T=T*.9;
//	    T=T*0.9;
	    best_errors.add(best_err);
	    count++;
	} else p=p-v;
	cerr <<"\n";
	if (!(count % 100) && (count>0)) cerr << "\n\nAnnealing iteration "<<count<<"\n\n\n";
    } while ((!have_first_err || (best_err > 0.0001 && count<2500)) && !amoeba_stop);
    DenseMatrix *dip_eval_out = new DenseMatrix(7,3);
    dip_eval_out->zero();
    (*dip_eval_out)[0][0]=(*dip_eval_out)[0][1]=(*dip_eval_out)[0][2]=best_p.x();
    (*dip_eval_out)[1][0]=(*dip_eval_out)[1][1]=(*dip_eval_out)[1][2]=best_p.y();
    (*dip_eval_out)[2][0]=(*dip_eval_out)[2][1]=(*dip_eval_out)[2][2]=best_p.z();
    (*dip_eval_out)[3][0]=(*dip_eval_out)[4][1]=(*dip_eval_out)[5][2]=1;
    dip_eval_oport->send(dip_eval_out);
    DenseMatrix *dip_vis_out = new DenseMatrix(2,6);
    (*dip_vis_out)[0][0]=(*dip_vis_out)[1][0]=best_p.x();
    (*dip_vis_out)[0][1]=(*dip_vis_out)[1][1]=best_p.y();
    (*dip_vis_out)[0][2]=(*dip_vis_out)[1][2]=best_p.z();
    (*dip_vis_out)[0][3]=(*dip_vis_out)[1][3]=cell_dir[best_idx].x();
    (*dip_vis_out)[0][4]=(*dip_vis_out)[1][4]=cell_dir[best_idx].y();
    (*dip_vis_out)[0][5]=(*dip_vis_out)[1][5]=cell_dir[best_idx].z();
    MatrixHandle voH(dip_vis_out);
    dip_vis_oport->send(MatrixHandle(voH));
    // gotta clear these ports so when we start again, everything works
    dir_inport->get(dir);
    rms_inport->get(rms);
    cerr << "Done annealing!  best_err="<<best_err<<"  best_p= ("<<best_idx<<") "<<best_p<<"\n    best_dir="<<cell_dir[best_idx]<<"  num_evals="<<num_evals<<"  count="<<count<<"\n";

    ColumnMatrix* cm=scinew ColumnMatrix(best_errors.size());
    for (int ii=0; ii<best_errors.size(); ii++) {
	(*cm)[ii]=best_errors[ii];
    }
    best_errors_oport->send(ColumnMatrixHandle(cm));
}

void Downhill_Simplex3::executeRoll() {
    cell_visited.resize(mesh_in->elems.size());
    cell_err.resize(mesh_in->elems.size());
    cell_dir.resize(mesh_in->elems.size());
    cell_visited.initialize(0);
    //cell_err;
    //cell_dir;
    //curr_idx;
    if (!mesh_in->have_all_neighbors) {
	mesh_in->compute_neighbors();
	mesh_in->compute_face_neighbors();
    }

    Point min, max;
    mesh_in->get_bounds(min, max);
    Vector diag(max-min);

    int curr_idx;
    Point p;
    do p=Point(min+Vector(diag.x()*mr(), diag.y()*mr(), diag.z()*mr()));
    while (!mesh_in->locate(p, curr_idx));
    
    int best_nbr;

    int have_first_err=0;
    int count=0;
    ColumnMatrixHandle rms;
    ColumnMatrixHandle dir;

    int num_evals=0;
    int cycle_idx=0;
    DenseMatrix dips(5,6);
    for (int i=0; i<5; i++) {
	dips[i][0]=p.x(); dips[i][1]=p.y();
	dips[i][2]=p.z(); dips[i][3]=dips[i][4]=0; dips[i][5]=0.001;
    }
    
    int foundBetter;
    Array1<int> nbrs;

    do {
	if (!mylock.tryLock()) {
	    mylock.lock();
	    mylock.unlock();
	} else {
	    mylock.unlock();
	}
	nbrs.resize(0);
	nbrs.add(curr_idx);
	mesh_in->get_elem_nbrhd(curr_idx, nbrs);
	int i;
	for (i=0; i<nbrs.size(); i++) {
	    int idx=nbrs[i];
	    Point p(mesh_in->elems[idx]->centroid());
//	    cerr << "Found p="<<p<<" in e="<<idx<<" ";
	    if (!useCache || !cell_visited[idx]) {
		cell_visited[idx]=1;
		DenseMatrix *dip_eval_out = new DenseMatrix(7,3);
		dip_eval_out->zero();
		int ii=0;
		if (useCache && cell_visited[idx]) 
		    for (; ii<3; ii++) (*dip_eval_out)[6][ii]=idx;
		(*dip_eval_out)[0][0]=(*dip_eval_out)[0][1]=(*dip_eval_out)[0][2]=p.x();
		(*dip_eval_out)[1][0]=(*dip_eval_out)[1][1]=(*dip_eval_out)[1][2]=p.y();
		(*dip_eval_out)[2][0]=(*dip_eval_out)[2][1]=(*dip_eval_out)[2][2]=p.z();
		(*dip_eval_out)[3][0]=(*dip_eval_out)[4][1]=(*dip_eval_out)[5][2]=1;
		dip_eval_oport->send_intermediate(dip_eval_out);

		// vis junk
		dips[4][0]=p.x();
		dips[4][1]=p.y();
		dips[4][2]=p.z();
		dips[4][3]=(dips[0][3]+dips[1][3]+dips[2][3]+dips[3][3])/3.;
		dips[4][4]=(dips[0][4]+dips[1][4]+dips[2][4]+dips[3][4])/3.;
		dips[4][5]=(dips[0][5]+dips[1][5]+dips[2][5]+dips[3][5])/3.;
		DenseMatrix *vis_out = new DenseMatrix(5,6);
		for (ii=0; ii<5; ii++) 
		    for (int jj=0; jj<6; jj++) 
			(*vis_out)[ii][jj]=dips[ii][jj];
		MatrixHandle voH(vis_out);
		dip_vis_oport->send_intermediate(MatrixHandle(voH));

		if(!dir_inport->get(dir)) {
		    cerr << "Error - Downhill_Simplex3 didn't get a dir!\n";
		    return;
		}
		if(!rms_inport->get(rms)) {
		    cerr << "Error - Downhill_Simplex3 didn't get a metric!\n";
		    return;
		}
		double *ee=rms->get_rhs();
		if (rms->nrows() != 1)
		    cerr << "Error - rms->nrows="<<rms->nrows()<<"\n";
		cell_err[idx]=ee[0];
		cell_dir[idx]=Vector(dir->get_rhs()[0], dir->get_rhs()[1],
				     dir->get_rhs()[2]);
		num_evals++;
		dips[cycle_idx][0]=p.x();
		dips[cycle_idx][1]=p.y();
		dips[cycle_idx][2]=p.z();
		dips[cycle_idx][3]=cell_dir[curr_idx].x();
		dips[cycle_idx][4]=cell_dir[curr_idx].y();
		dips[cycle_idx][5]=cell_dir[curr_idx].z();
		cycle_idx=(cycle_idx+1) % 4;
		count++;
	    }
//	    cerr << "e="<<cell_err[idx]<<"\n";
	}
	// find the smallest neighbor's error
	foundBetter=0;
	int best_idx=nbrs[0];
	double best_err=cell_err[curr_idx];
//	cerr << "ERRORS: ";
//	for (i=0; i<5; i++) {
//	    if (nbrs[i]<0) continue;
//	    cerr << i << " ("<<nbrs[i]<<") e="<<cell_err[nbrs[i]]<<"  ";
//	}
	cerr << "\n";
	for (i=1; i<nbrs.size(); i++) {
//	    if (nbrs[i]<0) continue;
	    if (cell_err[nbrs[i]] < best_err) {
		foundBetter=1;
		cerr << "FOUND BETTER!\n";
		best_idx = nbrs[i];
		best_err = cell_err[nbrs[i]];
	    }
	}
	curr_idx=best_idx;
    } while (foundBetter);

    int best_idx=curr_idx;
    Point best_p(mesh_in->elems[best_idx]->centroid());
    DenseMatrix *dip_eval_out = new DenseMatrix(7,3);
    dip_eval_out->zero();
    (*dip_eval_out)[0][0]=(*dip_eval_out)[0][1]=(*dip_eval_out)[0][2]=best_p.x();
    (*dip_eval_out)[1][0]=(*dip_eval_out)[1][1]=(*dip_eval_out)[1][2]=best_p.y();
    (*dip_eval_out)[2][0]=(*dip_eval_out)[2][1]=(*dip_eval_out)[2][2]=best_p.z();
    (*dip_eval_out)[3][0]=(*dip_eval_out)[4][1]=(*dip_eval_out)[5][2]=1;
    dip_eval_oport->send(dip_eval_out);
    DenseMatrix *dip_vis_out = new DenseMatrix(2,6);
    (*dip_vis_out)[0][0]=(*dip_vis_out)[1][0]=best_p.x();
    (*dip_vis_out)[0][1]=(*dip_vis_out)[1][1]=best_p.y();
    (*dip_vis_out)[0][2]=(*dip_vis_out)[1][2]=best_p.z();
    (*dip_vis_out)[0][3]=(*dip_vis_out)[1][3]=cell_dir[best_idx].x();
    (*dip_vis_out)[0][4]=(*dip_vis_out)[1][4]=cell_dir[best_idx].y();
    (*dip_vis_out)[0][5]=(*dip_vis_out)[1][5]=cell_dir[best_idx].z();
    MatrixHandle voH(dip_vis_out);
    dip_vis_oport->send(MatrixHandle(voH));
    // gotta clear these ports so when we start again, everything works
    dir_inport->get(dir);
    rms_inport->get(rms);
    cerr << "Done rolling!  best_err="<<cell_err[best_idx]<<"  best_p= ("<<best_idx<<") "<<best_p<<"\n    best_dir="<<cell_dir[best_idx]<<"  num_evals="<<num_evals<<"  count="<<count<<"\n";
}

void Downhill_Simplex3::execute() {
    if(!mesh_inport->get(mesh_in) || !mesh_in.get_rep()) {
	cerr << "Downhill_Simplex3 -- couldn't get mesh.  Returning.\n";
	return;
    }
    clString meth=methodTCL.get();
    useCache=useCacheTCL.get();
    if (meth=="downhill") {
	executeDownhill();
    } else if (meth=="anneal") {
	executeAnneal();
    } else if (meth=="protozoa") {
	executeProtozoa();
    } else {
	executeRoll();
    }
    cerr << "Done with the Module!"<<endl;
}


// helper is a second thread that runs the Numerical Recipes amoeba code
//   the main thread controls marshalling for port I/O, and startup/shoutdown 
// the two threads communicate through semaphores - pos_sem and error_sem
// the main thread waits on pos_sem - the amoeba has to tell it what position
//   it wants evaluated
// the helper thread waits on error_sem - it wants to know the error for
//   a new position, which the main thread evaluates by sending/receiving port
//   data
//---------------------------------------------------------------
void Downhill_Simplex3::helper(int /*proc*/){
    while(1){
	helper_sem->down();
	
	int i;
	if (use_protozoa) {
	  cerr <<"Calling prorozoa()"<<endl;
	  FTOL = 1.0e-7;
	  nfunc=200;
	  for (i=1;i<=ndim*2+1;i++) {
	    printf("%3d ",i);
	    for (int j=1;j<=ndim+3;j++) printf("%12.6f ",p[i][j]);
	    printf("%12.6f\n",y[i]);
	  }
	} else {
	  cerr <<"Calling amoeba()"<<endl;
	  FTOL = 1.0e-7;
	  nfunc=200;
	  for (i=1;i<=ndipoles+1;i++) {
	    printf("%3d ",i);
	    for (int j=1;j<=ndim+3;j++) printf("%12.6f ",p[i][j]);
	    printf("%12.6f\n",y[i]);
	  }
	}
	cerr <<"ndim = "<<ndim<<endl;

	amoeba_stop=0;
	if (use_protozoa) 
	  protozoa(p,y,ndim,FTOL,Downhill_Simplex3::error_eval,&nfunc,3,&amoeba_stop);
	else
	  amoeba(p,y,ndim,FTOL,Downhill_Simplex3::error_eval,&nfunc,3,&amoeba_stop);
	
	printf("\nNumber of function evaluations: %3d\n",nfunc);
	printf("function values at the vertices:\n\n");
	for (i=1;i<=ndipoles+1;i++) {
	    printf("%3d ",i);
	    for (int j=1;j<=ndim+3;j++) printf("%12.6f ",p[i][j]);
	    printf("%12.6f\n",y[i]);
	}
	
	send_pos=0;
	in_bounds = 1;
	need_error = 1;
	pos_sem->up();
    }
}

// this is the method that the amoeba calls when it needs to know the
//   error for a particular position.
// the "hidden" variables that are being used are "p", which is a 
//   matrix containing all of the dipoles and their current
//   positions/orientations; and "y" contains the error for each dipole in p
//---------------------------------------------------------------
double* Downhill_Simplex3::error_eval(int pi) {
    posIdx=pi;
    Point mypt(p[pi][1], p[pi][2], p[pi][3]);

    send_pos = 1;
    in_bounds = 1;
    need_error = 1;
    if (mesh_in->locate(mypt, curr_idx)) {
	if (!useCache || !cell_visited[curr_idx]) { // haven't evaluated this cell yet
	    cerr << "DS2 - haven't evaluated this cell yet...\n";
	    cell_visited[curr_idx] = 1;
	    pos_sem->up();
	    error_sem->down();
	    cell_err[curr_idx] = err[0];
	    cell_dir[curr_idx] = Vector(err[1], err[2], err[3]);
	} else {			     // already evaluated this cell
	    cerr << "DS2 - already evaluated this cell...\n";
	    need_error = 0;
	    pos_sem->up();
	    error_sem->down();
	}
    } else {
	cerr << "DS2 - point out of bounds...\n";
	in_bounds = 0;
	need_error = 0;
	pos_sem->up();
	error_sem->down();
    }
    return err;
}

void Downhill_Simplex3::tcl_command(TCLArgs& args, void* userdata) {
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
    } else if (args[1] == "print") {
        cerr << "Can't print - sorry.\n";
    } else if (args[1] == "stop") {
        amoeba_stop=1;
    } else {
        Module::tcl_command(args, userdata);
    }
}
} // End namespace DaveW
//---------------------------------------------------------------


