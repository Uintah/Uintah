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

// Submatrix solver for A * Phi = 0 system.
// s=scalp, v=volume, c=cortex
// Phi's are our x's -- potentials throughout system
// Phi_c = [Asv*(Avv)^(-1)*Avc-Asc]^(-1) * [Ass-Asv*(Avv)^(-1)*Avs] * Phi_s
// This time we will use sparse storage and sparse solvers (LU Decomposition)
// for the Avv inversion.  This should dramatically redulce the amount of
// space required, and the amount of time required for the problem -- since
// it represented over 80% of both with full matrix methods.


// The problem is stated as:  A * Phi = 0
// Where:      +-           -+
//             | Ass Asv Asc |      s = scalp
//     A =     | Avs Avv Avc |      v = volume
//             | Acs Acv Acc |      c = cortex
//             +-           -+
//             +-     -+
//             | Phi_s |
//   Phi =     | Phi_v |
//             | Phi_c |
//             +-     -+
// Phi_c = [Asv*(Avv)^(-1)*Avc-Asc]^(-1) * [Ass-Asv*(Avv)^(-1)*Avs] * Phi_s
// M = Asv * [(Avv)^(-1) * Avc] - Asc

/*
 *  SurfaceToSurface: Solve a linear system to determine cortical potentials
 *		 from scalp potentials.
 *		Input: matrix from BldFEMatrix (no BC's set, b/c none
 *		 were sent in from BldEEGMesh), and two column matrices
 *		 (also from BldEEGMesh) with Scalp and Cortex BC's --
 *		 used for BC values and to get size of submatrices right;
 *		 and SurfTree to map cortical potential onto at the end.
 *		Ouptut: SurfTree.
 *
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Packages/DaveW/ThirdParty/SparseLib/Leonid/Vector.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/SurfTree.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include <malloc.h>
//#include "string.h"

#include <Packages/DaveW/ThirdParty/OldLinAlg/matrix.h>
#include <Packages/DaveW/ThirdParty/OldLinAlg/vector.h>
#include <Packages/DaveW/ThirdParty/NumRec/dsvdcmp.h>
#include <Packages/DaveW/ThirdParty/NumRec/dsvbksb.h>
#include <Packages/DaveW/ThirdParty/NumRec/dpythag.h>
#include <Packages/DaveW/ThirdParty/OldLinAlg/cuthill.h>
#include <Packages/DaveW/ThirdParty/NumRec/bandec.h>
#include <Packages/DaveW/ThirdParty/NumRec/banbks.h>
#include <Packages/DaveW/ThirdParty/NumRec/banmul.h>
#include <Packages/DaveW/ThirdParty/NumRec/banmprv.h>

//#include <nrutil.h>
#define  COMPLEX std::complex<double>
//#undef   _NON_TEMPLATE_COMPLEX
//#include "/usr/include/CC/complex"
//#define _NON_TEMPLATE_COMPLEX
//#define COMPLEX complex<double>

//compressed row matrix storage
#include <comprow_double.h>
//matrix, vector input- output
#include <iotext_double.h>

//matrix definitions
#include <mvm.h>
//vector definitions
#include <mvv.h>
//vector multiplication BLASD
#include <mvblasd.h>


//#define _NON_TEMPLATE_COMPLEX
//preconditionars
#include <icpre_double.h>
#include <diagpre_double.h>
#include <ilupre_double.h>

//iterative IML  methods
#include <cg.h>
#include <bicg.h>
#include <qmr.h>
#include <cgs.h>
#include <bicgstab.h>
#include <ir.h>
#include <gmres.h>

#include <Packages/DaveW/ThirdParty/RegTools/reg_tools.h>

namespace BioPSE {
using namespace SCIRun;


void linbcg(unsigned long n, double b[], double x[], int itol, double tol,
	    int itmax, int *iter, double *err, double **a);


class SurfaceToSurface : public Module {
  MatrixIPort *imatrix;
  MeshIPort *imesh;
  SurfaceIPort* isurf;
  SurfaceOPort* osurf;
  MeshOPort* omesh;
  GuiString status;
  GuiInt maxiter;
  GuiDouble target_error;
  GuiInt iteration;
  GuiDouble current_error;
  CPUTimer timer;

  // these are globals for parallel code
  int *LUMap;	
  int *fwdMap;
  int *invMap;
  double **AvvU;
  double **AvvV;
  double **Avc;
  double **AvcTmp;
  int m1, m2, nv, nc;
  int np;
  Mutex mutex;
public:
  void parallel(int);
  SurfaceToSurface(GuiContext *context);
  virtual ~SurfaceToSurface();
  virtual void execute();
  void getBW(double **a, int n, int *m1, int *m2);
  void bldInvMap(int *fwdMap, int *invMap, int nv);
  void buildAc(double **Ac, int ns, int nc,
	       int nv, double **Avv, double **Avc, 
	       double **Asv, double **Asc);
  double rms(double *a, double *b, int n);
  double **splitMatrix(double **Ass, double **Asv, double **Asc, 
		       double **Avs, double **Avc, int ns, 
		       int nv, int nc, MatrixHandle mh);
  void setBdryCorticalBC(double *Phi_c, SurfTree *st, int start, int num);
  void solveSparse(int nr, int nc, double **A, double *x, double *b);
  void solveDense(int nr, int nc, double **A, double *x, double *b);
  void jacobi_sci(Matrix*,ColumnMatrix& , ColumnMatrix&);
};


DECLARE_MAKER(SurfaceToSurface)


SurfaceToSurface::SurfaceToSurface(GuiContext *context)
  : Module("SurfaceToSurface", context, Filter),
    status(context->subVar("status")),
    maxiter(context->subVar("maxiter")),
    target_error(context->subVar("target_error")),
    iteration(context->subVar("iteration")),
    current_error(context->subVar("current_error")),
    mutex("SurfaceToSurface mutex")
{
    imatrix=new MatrixIPort(this, "MatrixIn", MatrixIPort::Atomic);
    add_iport(imatrix);
    imesh=new MeshIPort(this, "MeshIn", MeshIPort::Atomic);
    add_iport(imesh);
    isurf=new SurfaceIPort(this, "SurfTreeIn", SurfaceIPort::Atomic);
    add_iport(isurf);

    // Create the output port
    osurf=new SurfaceOPort(this,"SurfTreeOut", SurfaceIPort::Atomic);
    add_oport(osurf);
    omesh=new MeshOPort(this,"MeshOut", MeshIPort::Atomic);
    add_oport(omesh);
}

SurfaceToSurface::~SurfaceToSurface()
{
}


void
SurfaceToSurface::setBdryCorticalBC(double *Phi_c, SurfTree *st,
				    int ns, int nc)
{
  for (int i=0; i<nc; i++)
  {
    st->data[i+ns]=Phi_c[i+1];
  }
  msgStream_ << "WRITING FROM "<<nc<<" to "<<nc+ns<<"\n";
}

    
void
SurfaceToSurface::getBW(double **a, int n, int *m1, int *m2)
{
  *m1=0;
  *m2=0;
  for (int i=1; i<=n; i++)
  {
    for (int j=1; j<=a[i][0]; j++)
    {
      int d=(int)a[i][j*2-1]-i;
      if (d>*m2) *m2=d;
      if (-d>*m1) *m1=-d;
    }
  }
}


void
SurfaceToSurface::bldInvMap(int *fwdMap, int *invMap, int nv)
{
  int i;
  for (i=1; i<=nv; i++)
  {
    invMap[i]=-123456;
  }
  for (i=1; i<=nv; i++)
  {
    invMap[fwdMap[i]]=i;
  }
}


void
SurfaceToSurface::parallel(int proc)
{
  int start_col=nc*proc/np+1;
  int end_col=nc*(proc+1)/np+1;

  double *t1 = makeVector(nv);
  for (int i=start_col; i<end_col; i++)
  {
    getColumn(Avc, nv, i, t1);
    remapVector(t1, invMap, nv);
    banbks(AvvU, nv, m1, m2, AvvV, LUMap, t1);
    remapVector(t1, fwdMap, nv);
    putColumn(AvcTmp, nv, i, t1);
  }
  freeVector(t1);
  mutex.lock();
  msgStream_ << "proc="<<proc<<" sc="<<start_col<<" ec="<<end_col<<"\n";
  mutex.unlock();
}


void
SurfaceToSurface::buildAc(double **Ac, int ns, int nc, 
			  int nv, double **Avv, double **Avc, 
			  double **Asv, double **Asc)
{
  int kk=0;
  int idx=0;
  double tt=0;
  for (int ll=1; ll<=nv; ll++)
  {
    tt+=Avv[ll][0];
    if (Avv[ll][0]>kk)
    {
      kk=(int)Avv[ll][0];
      idx=ll;
    }
  }
  msgStream_ <<"  STATS::: Most connected element Avv["<<idx<<"] has " << kk;
  msgStream_ << " connections.  Avg. has "<< tt/nv <<"\n";
  double **t = (double **) malloc (sizeof(double*)*(nv+1));
  int i;
  msgStream_ << "Starting Cuthill-McKee algorithm.\n";

  fwdMap = (int *) malloc (sizeof(int)*(nv+1));
  invMap = (int *) malloc (sizeof(int)*(nv+1));
  LUMap = (int *) malloc (sizeof(int)*(nv+1));
    
  //    printSparseMatrix(Avv,nv);
  int tries=cuthillMcKee(Avv, nv, fwdMap);
  bldInvMap(fwdMap, invMap, nv);
  msgStream_ << "Done with Cuthill-McKee.\n";

  for (i=1; i<=nv; i++)
  {
    t[i]=Avv[invMap[i]];
  }
  for (i=1; i<=nv; i++) 
  {
    Avv[i]=t[i];
  }
  for (i=1; i<=nv; i++)
  {
    for (int j=1; j<=Avv[i][0]; j++)
    {
      Avv[i][j*2-1]=fwdMap[(int)Avv[i][j*2-1]];
    }
  }

  getBW(Avv, nv, &m1, &m2);
  int bw=m1+m2+1;
  printf("m1(lower)=%d  m2(upper)=%d  totalBW=%d\n", m1, m2, bw);

  AvvU = makeMatrix(nv, bw);
  AvvV = makeMatrix(nv, m1);
  for (i=1; i<=nv; i++)
  {
    int j;
    for (j=1; j<=bw; j++)
    {
      AvvU[i][j]=0;
    }
    for (j=1; j<=m1; j++)
    {
      AvvV[i][j]=0;
    }
  }
  for (i=1; i<=nv; i++)
  {
    for (int j=1; j<=Avv[i][0]; j++)
    {
      AvvU[i][(int)Avv[i][j*2-1]-i+m1+1]=Avv[i][j*2];
    }
  }	
  double d;
  msgStream_ << "Starting SVD ("<<nv<<")...(timer="<<timer.time()<<") ";
  bandec(AvvU, nv, m1, m2, AvvV, LUMap, &d);
  msgStream_ << "Done!  (timer="<<timer.time()<<")\n";

  AvcTmp = makeMatrix(nv, nc);

  np=Thread::numProcessors();
  if (np>4) np/=2;	// being nice - just using half the processors. :)
  msgStream_ << "np="<<np<<"\n";
  msgStream_ << "Starting back substitution ("<<nc<<")...("<<timer.time()<<")... ";
  Thread::parallel(Parallel<SurfaceToSurface>(this, &SurfaceToSurface::parallel),
		   np, true);
  msgStream_ << "Done! (timer="<<timer.time()<<")\n";

  free(fwdMap);
  free(invMap);
  free(LUMap);
  freeMatrix(AvvV);
  freeMatrix(AvvU);

  double **AcTmp = makeMatrix(ns, nc); // to store Asv*AvvI*Avc

  msgStream_ << "Multiplying Asv * AvcTmp (to get AcTmp) (timer="<<timer.time()<<").\n";
  matMatMult(ns, nv, nc, Asv, AvcTmp, AcTmp);
	
  freeMatrix(AvcTmp);

  msgStream_ << "Subtracting AcTmp - Asc (to get Ac). (timer="<<timer.time()<<")\n";
  matSub(ns, nc, AcTmp, Asc, Ac);
}


double
SurfaceToSurface::rms(double *a, double *b, int n)
{
  double total=0;
  double diff=0;
  for(int i=1; i<=n; i++)
  {
    total+=a[i]*a[i];
    diff+=(a[i]-b[i])*(a[i]-b[i]);
  }
  return sqrt(diff/total)*100;
}


double **
SurfaceToSurface::splitMatrix(double **Ass, double **Asv, double **Asc, 
			      double **Avs, double **Avc, int ns, int nv, 
			      int nc, MatrixHandle mh)
{
  double **Avv;
  int i;
  for (i=1; i<=ns; i++)
  {	
    int j;
    for (j=1; j<=ns; j++)
    {	// Initialize Ass
      Ass[i][j]=0;
    }
    for (j=1; j<=nv; j++)
    {		// Initialize Asv and Avs
      Asv[i][j]=Avs[j][i]=0;
    }
    for (j=1; j<=nc; j++)
    {		// Initialize Asc
      Asc[i][j]=0;
    }
  }
  for (i=1; i<=nv; i++)
  {
    for (int j=1; j<=nc; j++)
    {		// Initialize Avc
      Avc[i][j]=0;
    }
  }
  int x, y;
  int lastS, lastV;
  lastS=ns;
  lastV=nv+lastS;
  double tt;
  Array1<int> idx;
  Array1<double> v;
  int maxvrow = 0;
  Array1<int> vrowlengths(nv+1);
  vrowlengths.initialize(0);
  int nn=ns+nv+nc;
  int xx;
  for (xx=0; xx<nn; xx++)
  {
    mh->getRowNonzeros(xx, idx, v);
    for (int yy=0; yy<idx.size(); yy++)
    {
      tt=v[yy];
      x=xx+1;
      y=idx[yy]+1;
      if (x<=lastS)
      {
	if (y<=lastS)
	{
	  Ass[x][y]=tt;
	}
	else if (y<=lastV)
	{
	  Asv[x][y-lastS]=Avs[y-lastS][x]=tt;
	}
	else
	{
	  Asc[x][y-lastV]=tt;
	}
      }
      else if (x<=lastV)
      {
	if (y<=lastS)
	{
	}
	else if (y<=lastV)
	{
	  int last = vrowlengths[x-lastS];
	  last++;
	  if (last>maxvrow) maxvrow=last;
	  vrowlengths[x-lastS]=last;
	}
	else
	{
	  Avc[x-lastS][y-lastV]=tt;
	}
      }
    }
  }

  Avv = makeMatrix(nv, maxvrow*2+2);
  msgStream_ << "nv="<<nv<<"  maxvrow="<<maxvrow<<"\n";
  for (i=1; i<=nv; i++)
    for (int j=0; j<maxvrow*2+2; j++)
      Avv[i][j]=0;

  for (xx=0; xx<nn; xx++)
  {
    mh->getRowNonzeros(xx, idx, v);
    for (int yy=0; yy<idx.size(); yy++)
    {
      tt=v[yy];
      x=xx+1;
      y=idx[yy]+1;
      if (x>lastS && x<=lastV &&y>lastS && y<=lastV)
      {
	int last=Avv[x-lastS][0];
	Avv[x-lastS][last*2+1]=y-lastS;
	Avv[x-lastS][last*2+2]=tt;
	Avv[x-lastS][0] = last+1;
      }
    }
  }
  printf("Done splitting matrix!\n");
  return Avv;
}


void
SurfaceToSurface::jacobi_sci(Matrix* matrix,
			     ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  int size=matrix->nrows();

  int flop=0;
  int memref=0;
  int gflop=0;
  int grefs=0;

  iteration.set(0);
    
  ColumnMatrix invdiag(size);
  // We should try to do a better job at preconditioning...
  int i;

  for(i=0;i<size;i++)
  {
    invdiag[i]=1./matrix->get(i,i);
  }
  flop+=size;
  memref=2*size*sizeof(double);

  ColumnMatrix Z(size);
  matrix->mult(lhs, Z, flop, memref);

  Sub(Z, Z, rhs, flop, memref);
  double bnorm=rhs.vector_norm(flop, memref);
  double err=Z.vector_norm(flop, memref)/bnorm;

  current_error.set(err);

  int niter=0;
  int toomany=maxiter.get();
  if(toomany == 0)
    toomany=2*size;
  double max_error=target_error.get();

  gflop+=flop/1000000000;
  flop=flop%1000000000;
  grefs+=memref/1000000000;
  memref=memref%1000000000;
    
  int last_update=0;

  Array1<int> targetidx;
  Array1<double> targetlist;
  targetidx.add(0);
  targetlist.add(max_error);

  double log_orig=log(err);
  double log_targ=log(max_error);
  while(niter < toomany)
  {
    niter++;

    double new_error;
    if(get_gui_doublevar(id, "target_error", new_error)
       && new_error != max_error)
    {
      targetidx.add(niter);
      targetlist.add(max_error);
      max_error=new_error;
    }
    targetidx.add(niter);
    targetlist.add(max_error);
    if(err < max_error)
      break;
    if(err > 10)
    {
      error("Solution not converging!");
      break;
    }

    Mult(Z, invdiag, Z, flop, memref);
    ScMult_Add(lhs, 1, lhs, Z, flop, memref);
    //	Sub(lhs, lhs, Z, flop, memref);

    matrix->mult(lhs, Z, flop, memref);
    Sub(Z, rhs, Z, flop, memref);
    err=Z.vector_norm(flop, memref)/bnorm;

    gflop+=flop/1000000000;
    flop=flop%1000000000;
    grefs+=memref/1000000000;
    memref=memref%1000000000;

    if(niter == 1 || niter == 5 || niter%10 == 0)
    {
      iteration.set(niter);
      current_error.set(err);
      double time=timer.time();

      double progress=(log_orig-log(err))/(log_orig-log_targ);
      update_progress(progress);
#if 0
      if(emit_partial.get() && niter%50 == 0)
	solport->send_intermediate(rhs.clone());
#endif
    }
  }
  iteration.set(niter);
  current_error.set(err);
}


#if 0
static void
writeit(SparseRowMatrix *matrix)
{
  TextPiostream stream("/tmp/my2.mat", Piostream::Write);
  Pio(stream, matrix);
}
#endif


// we'll build a Dataflow SparseRowMatrix and ColumnMatrix and solve them.
void
SurfaceToSurface::solveSparse(int nr, int ncol, double **AA, double *X, 
			      double *B)
{
    
  printSparseMatrix2("/tmp/my.mat", AA, nr);
  int nc=0;
  int i;
  for (i=1; i<=nr; i++) nc+=AA[i][0];
	
  int *rows = (int *) malloc ((nr+1)*sizeof(int));
  int *columns = (int *) malloc (nc * sizeof(int));
  double *data = (double *) malloc (nc * sizeof(double));

  int currcol=0;
    
  int rowtotal=0;

  for (i=0; i<nr; i++)
  {
    rows[i]=rowtotal;
    rowtotal += AA[i+1][0];
    for (int j=1; j<AA[i+1][0]*2; j+=2)
    {
      data[currcol]=AA[i+1][j+1];
      columns[currcol]=AA[i+1][j]-1;
      currcol++;
    }
  }
  rows[i]=rowtotal;

  msgStream_ << "Sorting columns,,,\n";
  for (int ii=0; ii<nr; ii++)
  {
    sortem2(&(columns[rows[ii]]), &(data[rows[ii]]), rows[ii+1]-rows[ii]);
  }

  msgStream_ << "Allocating sparse matrix...\n";
  SparseRowMatrix *matrix = new SparseRowMatrix(nr, ncol, rows, columns, 
						nc, data); 

  ColumnMatrix lhs(nr);
  for (i=0; i<nr; i++)
    lhs[i]=B[i+1];
  ColumnMatrix rhs(nr);

  int flag=1;

  // NOTE: this is cut and pasted from ../Matrix/SolveMatrix.cc lines 344-403
  // -------------------------------------

  int result; 
  int size = matrix->nrows();
  int non_zero =  matrix->get_row()[size];
    
  int maxit = maxiter.get();
  double tol = target_error.get();
  double x_init = 0.0;
    
  status.set("Running");
  TCL::execute("update idletasks");
  iteration.set(0);
  current_error.set(0);
  TCL::execute("update idletasks");
    
    
  VECTOR_double b(rhs.get_rhs(),size);
    
  VECTOR_double  x(size,x_init);
    
  for(i=0;i<size;i++)
    x[i] = lhs[i];
    
  CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 
    
    
  if (flag == 1)
  {
    DiagPreconditioner_double D(A);
    result = BiCG(A,x,b,D,maxit,tol);  //IML++ solver 
  }
    
  else if(flag == 2)
  {
    ICPreconditioner_double D(A);
    result = BiCG(A,x,b,D,maxit,tol);  //IML++ solver  
  }
    
  else if(flag ==3)
  {
    CompRow_ILUPreconditioner_double D(A);
    result = BiCG(A,x,b,D,maxit,tol);  //IML++ solver
  }
    
  for(i=0;i<size;i++)
    lhs[i] = x[i];
    
  if(result == 0)
    status.set("Done");
  else
    status.set("Failed to Converge");
    
  iteration.set(maxit);
  current_error.set(tol);    
  TCL::execute("update idletasks");
  // -------------------------------------

  double *xx=rhs.get_rhs();
  for (i=0; i<nr; i++)
    X[i+1]=xx[i];
}


// we'll build a Dataflow DenseMatrix and ColumnMatrix and solve them.
void
SurfaceToSurface::solveDense(int nr, int nc, double **A, double *x, 
			     double *b)
{
#if 0
  double **V = makeMatrix(nc, nc);
  double *W = makeVector(nc);

  int M=nr;
  int N=nc;
  int LDA=M;

  char JOBU='A';	// U vectors will overwrite A
  int LDU=M;
  double *U=new double[LDU*M];

    
  char JOBVT='A';
  int LDVT=N;
  double *VT=new double[LDVT*N];

  int LWORK = Max(3*Min(M,N)+Max(M,N),5*Min(M,N)-4);
  double *WORK=new double[LWORK];
  int INFO;

  //    double *A = new double[nr*nc];
  int count=0;
  for (int i=1; i<=nr; i++)
  {
    for (int j=1; j<=nc; j++)
    {
      //	    A[count]=A[i][j];
      count++;
    }
  }

  //    dgesvd_(&JOBU, &JOBVT, &M, &N, A->get_p(), &LDA, S, U, &LDU, VT, &LDVT, WORK,&LWORK, & INFO );

  //    dsvdcmp(A, nr, nc, W, V, 30);
  int trunc=0;
  for (i=1; i<nc; i++) 
    if (W[i] < 0.00001)
    {
      W[i]=0;
      trunc++;
    }
  msgStream_ << "nr = "<<nr<<"  nc = "<<nc<<"  trunc="<<trunc<<"\n";
  //    dsvbksb(A, W, V, nr, nc, b, x);
#endif

#if 0
  // gonna try to use a conjugate gradient solver here -- halt it when it
  // starts to go nuts... ;)
  double err;
  int iter;
  int nnew=Max(nr,nc);
  msgStream_ << "Allocating new (square) matrix...(timer="<<timer.time()<<")\n";
  double **squareA = makeMatrix(nnew, nnew);
  msgStream_ << "Copying into new (square) matrix...(timer="<<timer.time()<<")\n";
  double *xx = makeVector(nnew);
  double *bb = makeVector(nnew);
  for (int rr=1; rr<=nnew; rr++) 
    for (int cc=1; cc<=nnew; cc++)
    {
      double val;
      if (rr>nr) val=0;
      else if (cc>nc) val=0;
      else val=A[rr][cc];
      squareA[rr][cc] = val;
    }
  for (rr=1; rr<=nnew; rr++)
  {
    double val;
    if (rr>nr) val=0;
    else val=b[rr];
    bb[rr]=val;
  }
  for (int ll=1; ll<=nnew; ll++)
  {
    if (bb[ll] < -10000)
    {
      msgStream_ << "error in bb["<<ll<<"]\n";
    }
    for (int mm=1; mm<=nnew; mm++)
    {
      if (squareA[ll][mm] < -10000)
      {
	msgStream_ << "error in squareA["<<ll<<"]["<<mm<<"]\n";
      }
    }
  }
  msgStream_ << "Calling BiCG...(timer="<<timer.time()<<")\n";
  int maxiter = Max(nr,nc)*.8;
  msgStream_ << "   maxiter = "<<maxiter<<"\n";
  linbcg(nnew, bb, xx, 1, 0.0001, maxiter, &iter, &err, squareA);
  for (int cc=1; cc<=nc; cc++) x[cc]=xx[cc];	// copy back to sol'n
#endif

  // gonna try it with Leonid's TSVD
  double *matrix = new double[nr*nc];
  double *ptr = matrix;
  for (int c=0; c<nc; c++)
  {
    for (int r=0; r<nr; r++)
    {
      (*ptr) = A[r+1][c+1];
      ptr++;
    }
  }
  MatrixDense<double> AA(nr, nc, matrix);
  ZVector<double> bb(nr, &(b[1]));
  ZVector<double> xx(nc, &(x[1]));
  msgStream_ << xx <<"\n";

  //    LinearSystem<double> L1(AA,bb,"copy");
  //    L1.solve();
  //    L1.info();
  //    L1.print();
  //call tsvd with cut-off parameter 
  double cut_off = 0.001;
  tsvd(&AA,&bb,&xx,cut_off);  
    
  //cout<<"TSVD solution"<<endl;


  //    cout <<xx<<endl;
    
  //call dsvd with dumping 
  //    double lambda = 0.01;
  //    dsvd(&AA,&bb,&xx,lambda);  
    
  //    cout<<"DSVD solution"<<endl;
  //    cout <<xx<<endl;
}


void
SurfaceToSurface::execute()
{

  MeshHandle mesh;
  MatrixHandle mh;
  SurfaceHandle surfHandle;
  SurfTree *st;
  update_state(NeedData);

  if (!imatrix->get(mh)) return;
  if (!mh.get_rep())
  {
    error("Empty input matrix.");
    return;
  }
  if (!imesh->get(mesh)) return;
  if (!mesh.get_rep())
  {
    error("Empty input mesh.");
    return;
  }
  if (!isurf->get(surfHandle)) return;
  if (!surfHandle.get_rep())
  {
    error("Empty input surface.");
    return;
  }
  if (!(st=surfHandle->getSurfTree()))
  {
    error("Input surface wasn't s SurfTree.");
    return;
  }

  update_state(JustStarted);
  timer.clear();
  timer.start();
  int ns;
  for (ns=0; ns<mesh->nodes.size() && mesh->nodes[ns]->bc != 0; ns++);

  nc = st->idx.size()-ns;
  nv = mesh->nodes.size()-ns-nc;

  msgStream_ << "(ns,nv,nc)="<<ns<<" "<<nv<<" "<<nc<<"\n";

  update_progress(1,2);
  update_progress(2,2);

  // The problem is stated as:  A * Phi = 0
  // Where:      +-           -+
  //             | Ass Asv Asc |      s = scalp
  //     A =     | Avs Avv Avc |      v = volume
  //             | Acs Acv Acc |      c = cortex
  //             +-           -+
  //             +-     -+
  //             | Phi_s |
  //   Phi =     | Phi_v |
  //             | Phi_c |
  //             +-     -+
    
  // Solving for Phi_c as a function of Phi_s we derive:
  //        Phi_c = (Ac)^(-1) * As * Phi_s
  // where
  //        Ac = [(Avv)^(-1) * Avc] - [(Asv)^(-1) * Asc]    and
  //        As = [(Asv)^(-1) * Ass] - [(Avv)^(-1) * Avs]
  // read in Avv, Avc, Asv, Asc, Ass, Avs
  // reorder Avv w/ a Cuthill-McKee algorithm
  // build an LU decomposition of reordered Avv
  // plug in Avc one column at a time to build [(Avv)^(-1) * Avc] -- AvcTmp
  // plug in Avs one column at a time to build [(Avv)^(-1) * Avs] -- AvsTmp
  // Ac = Asv * [(Avv)^(-1) * Avc] - Asc
  // do a SVD on Ac, into Uc (in Ac memory), Wc, Vc
  // truncate Wc
  // build AcI (inv) from Wc, Vc, Ac (which is Uc) -- Ac^(-1)=V*W^(-1)*U(tr)
  // As = Ass - Asv * [(Avv)^(-1) * Avs]
  // Phi_c = (Asv * [(Avv)^(-1) * Avc] - Asc)^(-1) * 
  //         (Ass - Asv * [Avv^(-1) * Avs]) * Phi_s
  //       = AcI * As * Phi_s

  double **Avv;
  double **Asv = makeMatrix(ns, nv);
  double **Avs = makeMatrix(nv, ns);
  Avc = makeMatrix(nv, nc);
  double **Asc = makeMatrix(ns, nc);
  double **Ass = makeMatrix(ns, ns);
  double *Phi_c = makeVector(nc);
  double *Phi_s = makeVector(ns);
    
  msgStream_ << "Splitting matrix into submatrices...(time="<<timer.time()<<").\n";
  Avv=splitMatrix(Ass,Asv,Asc,Avs,Avc, ns, nv, nc, mh);

  //    printf("Here's Ass:\n");
  //    printMatrix(Ass, ns, ns);
  //    printf("Here's Asv:\n");
  //    printMatrix(Asv, ns, nv);
  //    printf("Here's Asc:\n");
  //    printMatrix(Asc, ns, nc);
  //    printf("Here's Avs:\n");
  //    printMatrix(Avs, nv, ns);
  //    printf("Here's Avv:\n");
  //    printSparseMatrix(Avv, nv);
  //    printf("Here's Avc:\n");
  //    printMatrix(Avc, nv, nc);
    
  msgStream_ << "\n\nns="<<ns<<"   nv="<<nv<<"  nc="<<nc<<"\n\n";


  msgStream_ << "Reading boundary conditions (time="<<timer.time()<<").\n";
  for (int ii=0; ii<ns; ii++) Phi_s[ii+1]=mesh->nodes[ii]->bc->value;

  double **Ac = makeMatrix(ns,nc);

  buildAc(Ac, ns, nc, nv, Avv, Avc, Asv, Asc);
  msgStream_ << "Done building Ac - timer="<<timer.time()<<"\n";
  // now we have Ac... which we'll call M

  double **M=Ac;

  double *q = makeVector(nv);
  double *r = makeVector(ns);
  double *s = makeVector(ns);
  double *t = makeVector(nv);
  double *u = makeVector(ns);

  // q = Avs * Phi_s   
  matVecMult(nv, ns, Avs, Phi_s, q);
    
  // t = Avv^(-1) * q
  msgStream_ << "Starting Solve sparse ("<<nv<<")...(time="<<timer.time()<<") ";
  solveSparse(nv, nv, Avv, t, q);  // Avv * t = q    
  msgStream_ << "Done!  (time="<<timer.time()<<")\n";

  // s = Asv * t
  matVecMult(ns, nv, Asv, t, s);

  // r = Ass * Phi_s
  matVecMult(ns, ns, Ass, Phi_s, r);

  // u = r - s;
  vectorSub(ns, r, s, u);
    
  // Phi_c = M^(-1) * u;
  msgStream_ << "Starting solve dense ("<<ns<<","<<nc<<")...(timer="
	     <<timer.time()<<") ";
  solveDense(ns, nc, M, Phi_c, u); // M * Phi_c = u
  msgStream_ << "Done!  (time="<<timer.time()<<")\n";

  msgStream_ << "going in...\n";
  setBdryCorticalBC(Phi_c, st, ns, nc);
  msgStream_ << "made it out!\n";

  osurf->send(surfHandle);

  int i;
  for (i=0; i<ns; i++) if (mesh->nodes[i]->bc == 0) msgStream_ << "WRONG!\n";
  for (i=0; i<ns; i++) mesh->nodes[i]->bc=0;
  for (i=ns+nv; i<ns+nv+nc; i++)
  {
    mesh->nodes[i]->bc = new DirichletBC(SurfaceHandle(0), 
					 Phi_c[i-ns-nv+1]);
    //if ((st->points[st->bcIdx[i-nv]]-mesh->nodes[i]->p).length2() > .000001)
    //   msgStream_ << "ERROR - not same node: surf="<<
    //   st->points[st->bcIdx[i-nv]]<<"  mesh="<<mesh->nodes[i]->p<<"!\n";
  }

  omesh->send(mesh);
  msgStream_ << "Done with SurfaceToSurface::execute (timer="
	     <<timer.time()<<")\n";
  timer.stop();
}

} // End namespace BioPSE



