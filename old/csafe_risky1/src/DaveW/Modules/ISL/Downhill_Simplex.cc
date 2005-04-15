/****************************************************************
 *  Simple "Downhill_Simplex module"for the SCIRun              *
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

#include <DaveW/ThirdParty/NumRec/amoeba.h>
#include <DaveW/ThirdParty/NumRec/nrutil.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/SparseRowMatrix.h>
#include <SCICore/Datatypes/SymSparseRowMatrix.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Thread.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <math.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::TclInterface;

using SCICore::Thread::Mutex;
using SCICore::Thread::Parallel;
using SCICore::Thread::Semaphore;
using SCICore::Thread::Thread;

int ndim;
int send_pos; 
double* err;
Semaphore *error_sem, *pos_sem, *helper_sem;

class Downhill_Simplex : public Module {    
    MatrixIPort* dipole_inport;
    ColumnMatrixIPort* rms_inport;
    ColumnMatrixIPort* dir_inport;
    MatrixOPort* d_outport;
    MatrixOPort* dipole_outport;
    
    int ndipoles; 
    double FTOL;
    int nfunc;
    double ** p; //holds initial dipole configuration 
    double* y; //holds initial errors for p configuration
    int counter;
    clString state;
    int first;
public:
    Mutex mylock;
    TCLstring tcl_status;
    Downhill_Simplex(const clString& id);
    virtual ~Downhill_Simplex();
    virtual void execute();
    static double *error_eval(double *);
    void helper(int proc);
    void tcl_command( TCLArgs&, void * );
}; //class


extern "C" Module* make_Downhill_Simplex(const clString& id) {
    return new Downhill_Simplex(id);
}

//---------------------------------------------------------------
Downhill_Simplex::Downhill_Simplex(const clString& id)
: Module("Downhill_Simplex", id, Filter), tcl_status("tcl_status",id,this),
  mylock("pause lock for Downhill_Simplex")
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
    
    d_outport = new MatrixOPort(this, "DipolePos Out port",
				MatrixIPort::Atomic);
    add_oport(d_outport);
//    dipoley_outport = new ColumnMatrixOPort(this, "DipoleY Out port",
//					    ColumnMatrixIPort::Atomic);
//    add_oport(dipoley_outport);
//    dipolez_outport = new ColumnMatrixOPort(this, "DipoleZ Out port",
//					    ColumnMatrixIPort::Atomic);
//    add_oport(dipolez_outport);

    dipole_outport = new MatrixOPort(this, "DipoleMat Out port",
				    MatrixIPort::Atomic);
    add_oport(dipole_outport);
    
    counter = 0;
    state = "START";
    ndim = 3;
    ndipoles = (ndim + 1);
    
    error_sem = new Semaphore("Downhill_Simplex error sync", 0);
    pos_sem = new Semaphore("Downhill_Simplex position sync", 0);
    helper_sem = new Semaphore("Downhill_Simplex amoeba sync", 0);
    
    Thread::parallel(Parallel<Downhill_Simplex>(this, &Downhill_Simplex::helper), 1, false);
//    Task::multiprocess(1, start_me_up, this, false);
    
    cerr<<"Constructor Done!"<<endl;
    mylock.unlock();
    first=1;
}

//------------------------------------------------------------
Downhill_Simplex::~Downhill_Simplex(){}

//--------------------------------------------------------------

void Downhill_Simplex::execute() {
    MatrixHandle dipole_in;
    if(!dipole_inport->get(dipole_in)) {
	cerr << "Downhill_simplex -- couldn't get dipoles.  Returning.\n";
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
    
#if 0
    if (!first) {
	ColumnMatrixHandle rms;
	ColumnMatrixHandle dir;
	dir_inport->get(dir);
	rms_inport->get(rms);
	first=0;
    }
#endif

    p = dmatrix(1,ndipoles+1,1,ndim+3);
    y = dvector(1,ndim);
    
    send_pos = 1;
    int i;
    for (i=0; i<ndipoles; i++) {
	for (int j=0; j<3; j++) {
	    p[i+1][j+1]=(*dipole_in.get_rep())[i][j];
	}
	p[i+1][4]=p[i+1][5]=0;
	p[i+1][6]=1;
    }
    p[i+1][1]=p[i+1][2]=p[i+1][3]=p[i+1][4]=p[i+1][5]=p[i+1][6]=0;
    
    DenseMatrix* dipole_out = new DenseMatrix(6,3);
    dipole_out->zero();
    for (int ii=0; ii<3; ii++)
	for (int jj=0; jj<3; jj++)
	    (*dipole_out)[ii][jj]=p[1][ii+1];

    cerr << "Downhill_Simplex: about to do a send_intermediate...\n";
    d_outport->send_intermediate(dipole_out);
    cerr << "Downhill_Simplex: sent intermediate!\n";

    cerr << "dipole_out=("<<(*dipole_out)[0][0]<<", "<<(*dipole_out)[1][0]<<", "<<(*dipole_out)[2][0]<<")\n";


    
    counter=1;
    while(1) {
//	cerr << "ABOUT TO TRY LOCK\n";
	if (!mylock.tryLock()) {
//	    cerr << "Downhill_simplex -- Pausing...\n";
	    mylock.lock();
	    mylock.unlock();
//	    cerr << "Downhill_simplex -- Unpausing...\n";
	} else {
//	    cerr << "Downhill_simplex -- got the lock, but I don't want it...";
	    mylock.unlock();
//	    cerr << " so I just unlocked it.\n";
	}
//	cerr << "DONE TRYING LOCK\n";
	
//	cerr <<"Getting new error!"<<endl;
	ColumnMatrixHandle rms;
	ColumnMatrixHandle dir;
	MatrixHandle dipole_in;

#if 0
	cerr << "Downhill_Simplex: about to read in matrix...\n";

	if(!dipole_inport->get(dipole_in)) {
	    cerr << "Error - Downhill_Simplex didn't get a dipole!\n";
	    return; 
	}

	cerr << "Downhill_Simple: got matrix!\n";
#endif

//	cerr << "Downhill_Simplex: about to read in ports...\n";
	if(!dir_inport->get(dir)) {
	    cerr << "Error - Downhill_Simplex didn't get a direction!\n";
	    return;
	}
//	cerr << "Downhill_Simplex: got direction.\n";
	if(!rms_inport->get(rms)) {
	    cerr << "Error - Downhill_Simplex didn't get an errorMetric!\n";
	    return;
	}
//	cerr << "Downhill_Simplex: got errormetric.\n";
	
//	cerr << "Downhill_Simplex -- got the data!!\n";
	double *rms_err = rms->get_rhs();
	double *dirP=dir->get_rhs();
	
	err=new double[4];
	err[0]=rms_err[0]; 
	err[1]=dirP[0]; err[2]=dirP[1]; err[3]=dirP[2];
	cerr << "Downhill_Simplex: Direction=(" << err[1] << "," << err[2] << "," << err[3];
	cerr << " ) - error=" << err[0] << "\n";
	
	if (counter>ndipoles) {
	    error_sem->up();
	}
	
	dipole_out = new DenseMatrix(6, 3);
	dipole_out->zero();
	
	int ii=0;
	if (counter < ndipoles) {
	    y[counter]=*err;
	    for (; ii<3; ii++) 
		for (int jj=0; jj<3; jj++) 
		    (*dipole_out)[ii][jj]=p[counter+1][ii+1];
	    for (; ii<6; ii++) p[counter][ii+1]=dirP[ii-3];
	    counter++;
	} else {
	    for (ii=3; ii<6; ii++) p[counter][ii+1]=dirP[ii-3];
	    if (counter == ndipoles) {
		y[counter]=*err;
		for (ii=0; ii<3; ii++)
		    for (int jj=0; jj<3; jj++) 
			(*dipole_out)[ii][jj]=p[counter+1][ii+1];
		helper_sem->up();
		counter++;
	    }
	    pos_sem->down();
	}	
	
	if (counter>ndipoles) 
	    for (ii=0; ii<3; ii++) 
		for (int jj=0; jj<3; jj++)
		    (*dipole_out)[ii][jj]=p[ndipoles+1][ii+1];
	
	DenseMatrix* dips = new DenseMatrix(ndipoles+1, ndim+3);
	MatrixHandle dipoles(dips);

//	cerr << "Here are the dipoles (straight from p):\n";
	for (ii=0; ii<ndipoles+1; ii++) {
//	    cerr << "   "<<ii+1<<" = ";
	    for (int jj=0; jj<ndim+3; jj++) {
		(*dips)[ii][jj]=p[ii+1][jj+1];
//		cerr << p[ii+1][jj+1] << " ";
	    }
//	    cerr << "\n";
	}

	if (send_pos) {
	    d_outport->send_intermediate(dipole_out);
	    dipole_outport->send_intermediate(dipoles);
	} else {
	    cerr << "I'm sending final version\n";
	    d_outport->send(dipole_out);
	    dipole_outport->send(dipoles);
	    // gotta clear these ports so when we start again, everything works
	    dir_inport->get(dir);
	    rms_inport->get(rms);
	    break;
	}
    }
    cerr << "Done with the Module!"<<endl;
}
//---------------------------------------------------------------
void Downhill_Simplex::helper(int /*proc*/){
    while(1){
	helper_sem->down();
	
	cerr <<"Calling amoeba()"<<endl;
	FTOL = 1.0e-7;
	nfunc=200;
	
	int i;
	for (i=1;i<=ndipoles+1;i++) {
	    printf("%3d ",i);
	    for (int j=1;j<=ndim+3;j++) printf("%12.6f ",p[i][j]);
	    printf("%12.6f\n",y[i]);
	}
	
	cerr <<"ndim = "<<ndim<<endl;
	
	amoeba(p,y,ndim,FTOL,Downhill_Simplex::error_eval,&nfunc,3);
	
	printf("\nNumber of function evaluations: %3d\n",nfunc);
	printf("function values at the vertices:\n\n");
	for (i=1;i<=ndipoles+1;i++) {
	    printf("%3d ",i);
	    for (int j=1;j<=ndim+3;j++) printf("%12.6f ",p[i][j]);
	    printf("%12.6f\n",y[i]);
	}
	
	send_pos=0;
	pos_sem->up();
    }
}
//---------------------------------------------------------------
double* Downhill_Simplex::error_eval(double *) {
    send_pos = 1;
    pos_sem->up();
    error_sem->down();
    return err;
}

void Downhill_Simplex::tcl_command(TCLArgs& args, void* userdata) {
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
    } else {
        Module::tcl_command(args, userdata);
    }
}
//---------------------------------------------------------------
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.4  2000/03/17 09:25:47  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/10/07 02:06:37  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/09/05 23:16:46  dmw
// new module
//
// Revision 1.1  1999/09/02 04:50:04  dmw
// more of Dave's modules
//
//
