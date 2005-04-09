
#include <Classlib/Pstreams.h>
#include <Classlib/Timer.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/SymSparseRowMatrix.h>
#include <Multitask/Task.h>
#include <iostream.h>
#include <math.h>

void usage(char* progname)
{
    cerr << "usage: " << progname << " filebase [nprocessors]" << endl;
    exit(1);
}

void serial_conjugate_gradient(Matrix* matrix,
			ColumnMatrix& lhs, ColumnMatrix& rhs)
{
    CPUTimer timer;
    timer.start();
    int size=matrix->nrows();

    int flop=0;
    int memref=0;
    int gflop=0;
    int grefs=0;
    
    ColumnMatrix diag(size);
    // We should try to do a better job at preconditioning...
    int i;

    for(i=0;i<size;i++){
	diag[i]=1./matrix->get(i,i);
    }
    flop+=size;
    memref=2*size*sizeof(double);

    ColumnMatrix R(size);
    matrix->mult(lhs, R, flop, memref);


    Sub(R, rhs, R, flop, memref);
    double bnorm=rhs.vector_norm(flop, memref);

    ColumnMatrix Z(size);
    matrix->mult(R, Z, flop, memref);

    ColumnMatrix P(size);
    double bkden=0;
    double err=R.vector_norm(flop, memref)/bnorm;
    if(err == 0){
	lhs=rhs;
	return;
    }

    int niter=0;
    int toomany=0;
    if(toomany == 0)
	toomany=2*size;
    double max_error=1.e-4;

    double time=timer.time();
    gflop+=flop/1000000000;
    flop=flop%1000000000;
    grefs+=memref/1000000000;
    memref=memref%1000000000;
    
    Array1<double> errlist;
    errlist.add(err);
    int last_update=0;

    Array1<int> targetidx;
    Array1<double> targetlist;
    int last_errupdate=0;
    targetidx.add(0);
    targetlist.add(max_error);

    double log_orig=log(err);
    double log_targ=log(max_error);
    while(niter < toomany){
	niter++;

	if(err < max_error)
	    break;

	// Simple Preconditioning...
	Mult(Z, R, diag, flop, memref);	

	// Calculate coefficient bk and direction vectors p and pp
	double bknum=Dot(Z, R, flop, memref);

	if(niter==1){
	    P=Z;
	    memref+=2*sizeof(double);
	} else {
	    double bk=bknum/bkden;
	    ScMult_Add(P, bk, P, Z, flop, memref);
	}
	bkden=bknum;

	// Calculate coefficient ak, new iterate x and new residuals r and rr
	matrix->mult(P, Z, flop, memref);

	double akden=Dot(Z, P, flop, memref);

	double ak=bknum/akden;
	ScMult_Add(lhs, ak, P, lhs, flop, memref);
	ScMult_Add(R, -ak, Z, R, flop, memref);

	err=R.vector_norm(flop, memref)/bnorm;

	errlist.add(err);

	gflop+=flop/1000000000;
	flop=flop%1000000000;
	grefs+=memref/1000000000;
	memref=memref%1000000000;

	if(niter%10 == 0){
	    cerr << "iteration: " << niter << ", error=" << err << endl;
	}
    }
    time=timer.time();
    cerr << "FLOPS: " << gflop*1.e9+flop << endl;
    cerr << "FLOP rate: " << (gflop*1.e3+flop*1.e-6)/time << " Mflops" << endl;
    cerr << "Memrefs: " << grefs*1.e9+memref << endl;
    cerr << "Mem bandwidth: " << (grefs*1.e3+memref*1.e-6)/time << " MB/sec" << endl;
    timer.stop();
    cerr << "Solved in " << timer.time() << " seconds, using " << niter << " iterations" << endl;
}

struct Result {
    double r;
    double pad[15];
};    

struct CGData {
    ColumnMatrix* rhs;
    ColumnMatrix* lhs;
    Matrix* mat;
    ColumnMatrix* diag;
    int niter;
    int toomany;
    ColumnMatrix* Z;
    ColumnMatrix* R;
    ColumnMatrix* P;
    double max_error;
    Barrier barrier;
    int np;
    Result* res1;
    Result* res2;
    Result* res3;
    double err;
    double bnorm;
};

void parallel_conjugate_gradient(void* d, int processor)
{
    CGData* data=(CGData*)d;
    WallClockTimer timer;
    Matrix* matrix=data->mat;
    int size=matrix->nrows();

    int flop=0;
    int memref=0;
    int gflop=0;
    int grefs=0;
    int beg=processor*size/data->np;
    int end=(processor+1)*size/data->np;

    if(processor==0){
	timer.start();
	data->diag=new ColumnMatrix(size);
	// We should try to do a better job at preconditioning...
	int i;

	for(i=0;i<size;i++){
	    ColumnMatrix& diag=*data->diag;
	    diag[i]=1./matrix->get(i,i);
	}
	flop+=size;
	memref=2*size*sizeof(double);

	data->R=new ColumnMatrix(size);
	ColumnMatrix& R=*data->R;
	ColumnMatrix& lhs=*data->lhs;
	matrix->mult(lhs, R, flop, memref);


	ColumnMatrix& rhs=*data->rhs;
	Sub(R, rhs, R, flop, memref);
	data->bnorm=rhs.vector_norm(flop, memref);

	data->Z=new ColumnMatrix(size);
	ColumnMatrix& Z=*data->Z;
	matrix->mult(R, Z, flop, memref);

	data->P=new ColumnMatrix(size);
	ColumnMatrix& P=*data->P;
	data->err=R.vector_norm(flop, memref)/data->bnorm;
	if(data->err == 0){
	    lhs=rhs;
	    return;
	}

	data->niter=0;
	data->toomany=0;
	if(data->toomany == 0)
	    data->toomany=2*size;
	data->max_error=1.e-4;
	data->res1=new Result[data->np];
	data->res2=new Result[data->np];
	data->res3=new Result[data->np];

	gflop+=flop/1000000000;
	flop=flop%1000000000;
	grefs+=memref/1000000000;
	memref=memref%1000000000;
    }
    data->barrier.wait(data->np);
    double err=data->err;
    double bkden=0;
    while(data->niter < data->toomany){
	if(err < data->max_error)
	    break;

	ColumnMatrix& Z=*data->Z;
	ColumnMatrix& P=*data->P;
	if(processor==0){
	    data->niter++;
	}

	    // Simple Preconditioning...
	ColumnMatrix& diag=*data->diag;
	ColumnMatrix& R=*data->R;
	Mult(Z, R, diag, flop, memref, beg, end);	

	// Calculate coefficient bk and direction vectors p and pp
	data->res1[processor].r=Dot(Z, R, flop, memref, beg, end);
	data->barrier.wait(data->np);

	double  bknum=0;
	for(int ii=0;ii<data->np;ii++)
	    bknum+=data->res1[ii].r;

	if(data->niter==1){
	    Copy(P, Z, flop, memref, beg, end);
	} else {
	    double bk=bknum/bkden;
	    ScMult_Add(P, bk, P, Z, flop, memref, beg, end);
	}
	data->barrier.wait(data->np);
	// Calculate coefficient ak, new iterate x and new residuals r and rr
	matrix->mult(P, Z, flop, memref, beg, end);
	bkden=bknum;
	data->res2[processor].r=Dot(Z, P, flop, memref, beg, end);
	data->barrier.wait(data->np);

	double akden=0;
	for(ii=0;ii<data->np;ii++)
	    akden+=data->res2[ii].r;

	double ak=bknum/akden;
	ColumnMatrix& lhs=*data->lhs;
	ScMult_Add(lhs, ak, P, lhs, flop, memref, beg, end);
	ColumnMatrix& rhs=*data->rhs;
	ScMult_Add(R, -ak, Z, R, flop, memref, beg, end);

	data->res3[processor].r=R.vector_norm(flop, memref, beg, end)/data->bnorm;
	data->barrier.wait(data->np);
	err=0;
	for(ii=0;ii<data->np;ii++)
	    err+=data->res3[ii].r;
	gflop+=flop/1000000000;
	flop=flop%1000000000;
	grefs+=memref/1000000000;
	memref=memref%1000000000;

	if(processor == 0 && data->niter%10 == 0){
	    cerr << "iteration: " << data->niter << ", error=" << err << endl;
	}
    }
    if(processor==0){
	data->niter++; // An extra one, since we leave the loop earlier
	double time=timer.time();
	cerr << "FLOPS: " << gflop*1.e9+flop << endl;
	cerr << "FLOP rate: " << (gflop*1.e3+flop*1.e-6)/time << " Mflops" << endl;
	cerr << "Memrefs: " << grefs*1.e9+memref << endl;
	cerr << "Mem bandwidth: " << (grefs*1.e3+memref*1.e-6)/time << " MB/sec" << endl;
	timer.stop();
	cerr << "Solved in " << timer.time() << " seconds, using " << data->niter << " iterations" << endl;
    }
}

extern "C" {
     void PSLDLT_Preprocess (
         int token,
         int n,
         int pointers[],
         int indices[],
         int *nonz,
         double *ops
         );

     void PSLDLT_Factor (
         int token,
         int n,
         int pointers[],
         int indices[],
         double values[]
         );

     void PSLDLT_Solve (
         int token,
         double x[],
         double b[]
         );

     void PSLDLT_Destroy (
         int token
         );
 };

main(int argc, char** argv)
{
    Task::initialize(argv[0]);
    if(argc != 2 && argc != 3)
	usage(argv[0]);
    int np;
    if(argc == 3)
	np=atoi(argv[2]);
    else
	np=Task::nprocessors();
    char buf[200];
    strcpy(buf, argv[1]);
    strcat(buf, ".mat");
    Piostream* stream=auto_istream(buf);
    if(!stream){
	cerr << "Error opening matrix: " << buf << endl;
	exit(1);
    }
    MatrixHandle mat;
    Pio(*stream, mat);
    if(!mat.get_rep()){
	cerr << "Error reading matrix: " << buf << endl;
	exit(1);
    }
    delete stream;

    strcpy(buf, argv[1]);
    strcat(buf, ".rhs");
    stream=auto_istream(buf);
    if(!stream){
	cerr << "Error opening rhs: " << buf << endl;
	exit(1);
    }
    ColumnMatrixHandle rhshandle;
    Pio(*stream, rhshandle);
    if(!rhshandle.get_rep()){
	cerr << "Error reading rhs: " << buf << endl;
	exit(1);
    }

    strcpy(buf, argv[1]);
    strcat(buf, ".sol");
    stream=auto_istream(buf);
    if(!stream){
	cerr << "Error opening solution: " << buf << endl;
	exit(1);
    }
    ColumnMatrixHandle correct_solhandle;
    Pio(*stream, correct_solhandle);
    if(!correct_solhandle.get_rep()){
	cerr << "Error reading solution: " << buf << endl;
	exit(1);
    }

    ColumnMatrixHandle solhandle(new ColumnMatrix(rhshandle->nrows()));
    ColumnMatrix& rhs=*rhshandle.get_rep();
    ColumnMatrix& sol=*solhandle.get_rep();
    ColumnMatrix& correct_sol=*correct_solhandle.get_rep();
    cerr << "Will use " << np << " processors " << endl;
#if 1
    if(np==1){
	serial_conjugate_gradient(mat.get_rep(), sol, rhs);
    } else {
	CGData* data=new CGData;
	data->np=np;
	data->rhs=rhshandle.get_rep();
	data->lhs=solhandle.get_rep();
	data->mat=mat.get_rep();
	Task::multiprocess(data->np, parallel_conjugate_gradient, data);
    }
#else
    WallClockTimer timer;
    timer.start();
    int token=4333;
    SymSparseRowMatrix* smat=(SymSparseRowMatrix*)mat.get_rep();
    int n=smat->nrows();
    int* pointers=smat->upper_rows;
    int* indices=smat->upper_columns;
    double* values=smat->upper_a;
    int nonz;
    double ops;
    PSLDLT_Preprocess(token, n, pointers, indices, &nonz, &ops);
    double* b=&rhs[0];
    double* x=&sol[0];
    cerr << "Preprocess time: " << timer.time() << endl;
    PSLDLT_Factor(token, n, pointers, indices, values);
    cerr << "Factor time: " << timer.time() << endl;
    PSLDLT_Solve(token, x, b);
    cerr << "Solve time: " << timer.time() << endl;
    PSLDLT_Destroy(token);
    timer.stop();
    cerr << "finished in " << timer.time() << " seconds\n";
#endif
    Sub(correct_sol, correct_sol, sol);
    double norm=correct_sol.vector_norm();
    cerr << "Error is " << norm << endl;
    strcpy(buf, argv[1]);
    strcat(buf, ".out");
    BinaryPiostream out(buf, Piostream::Write);
    Pio(out, solhandle);
    Task::exit_all(0);
}
