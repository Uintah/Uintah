/*
 *  Tikhonov.cc:
 *
 *  Written by:
 *   oleg
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Packages/MatlabInterface/share/share.h>

#include <stdio.h>

void tikhonov(double *f,double *d,double *r,double *w,double *m,
              double noise,double mu,double ml,int Nd,int Nm);

namespace MatlabInterface {

using namespace SCIRun;

class MatlabInterfaceSHARE Tikhonov : public Module 
{
  GuiString hpTCL;
  MatrixIPort *iport1;
  MatrixIPort *iport2;
  MatrixOPort *oport1;
  MatrixOPort *oport2;

public:
  Tikhonov(GuiContext *context);
  virtual ~Tikhonov();
  virtual void execute();
};


DECLARE_MAKER(Tikhonov)


Tikhonov::Tikhonov(GuiContext *context)
  : Module("Tikhonov", context, Filter, "Math", "MatlabInterface"),
    hpTCL(context->subVar("hpTCL"))
{
}


Tikhonov::~Tikhonov()
{
}


void
Tikhonov::execute()
{
// DECLARATIONS

  const char *gui; double noise; float tmp1;
  double *F,*d,*m,*r,*w;
  int    Nd,Nm;

  MatrixHandle mh1,mh2,mh3,mh4;

  iport1 = (MatrixIPort *)get_iport("Lead Field (Sensitivity) Matrix");
  iport2 = (MatrixIPort *)get_iport("RHS (data)");
  oport1 = (MatrixOPort *)get_oport("Resulting Model (Sources)");
  oport2 = (MatrixOPort *)get_oport("Residual");

  if (!iport1) {
    error("Unable to initialize iport 'Lead Field (Sensitivity) Matrix'.");
    return;
  }
  if (!iport2) {
    error("Unable to initialize iport 'RHS (data)'.");
    return;
  }
  if (!oport1) {
    error("Unable to initialize oport 'Resulting Model (Sources)'.");
    return;
  }
  if (!oport2) {
    error("Unable to initialize oport 'Residual'.");
    return;
  }
  
  DenseMatrix  *inp1;   // Sensitivity matrix
  ColumnMatrix *inp2;   // data (right-hand side)
  ColumnMatrix *otp1;   // model
  ColumnMatrix *otp2;   // residual
  ColumnMatrix *wwww;   // weighting matrix

// OBTAIN SCALAR PARAMETERS FROM GUI

  gui=hpTCL.get().c_str();
  sscanf(gui,"%g",&tmp1);
  noise=(double)tmp1;

// OBTAIN F FROM FIRST INPUT PORT

  iport1->get(mh1);
  inp1=dynamic_cast<DenseMatrix*>(mh1.get_rep()); //upcast
  if(inp1==NULL)
  {
    fprintf(stderr,"Focusing needs DenseMatrix as first input\n");
    return;
  }
  F=(double*)&((*inp1)[0][0]);

// OBTAIN PROBLEM DIMENSIONS

  Nd=inp1->nrows();
  Nm=inp1->ncols();

// OBTAIN d FROM SECOND INPUT PORT

  iport2->get(mh2);
  inp2=dynamic_cast<ColumnMatrix*>(mh2.get_rep()); //upcast
  if(inp2==NULL)
  {
    fprintf(stderr,"Focusing needs ColumnMatrix as second input\n");
    return;
  }
  d=(double*)&((*inp2)[0]);

// CREATE m

  otp1=scinew ColumnMatrix(Nm);
  mh3=MatrixHandle(otp1);
  m=&((*otp1)[0]);

// CREATE r

  otp2=scinew ColumnMatrix(Nd);
  mh4=MatrixHandle(otp2);
  r=&((*otp2)[0]);

// CREATE w

  wwww=scinew ColumnMatrix(Nm);
  w=&((*wwww)[0]);
  for(int i=0;i<Nm;i++) w[i]=1.;

// ACTUAL OPERATION (EMULATOR FOR NOW)

  tikhonov(F,d,r,w,m,noise,1e5,-1e+5,Nd,Nm);

  // fprintf(stderr,"Noise = %g\n",noise);
  // for(int i=0;i<Nm;i++) m[i]=F[i];
  // for(int i=0;i<Nd;i++) r[i]=d[i];

// SEND RESULTS DOWNSTREAM

  oport1->send(mh3);
  oport2->send(mh4);
}


} // End namespace MatlabInterface


