/*
 *  Focusing.cc:
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

namespace MatlabInterface {

using namespace SCIRun;

class MatlabInterfaceSHARE Focusing : public Module 
{
  GuiString hpTCL;
  MatrixIPort *iport1;
  MatrixIPort *iport2;
  MatrixOPort *oport1;
  MatrixOPort *oport2;

public:
  Focusing(const string& id);
  virtual ~Focusing();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MatlabInterfaceSHARE Module* make_Focusing(const string& id) {
  return scinew Focusing(id);
}

Focusing::Focusing(const string& id)
  : Module("Focusing", id, Filter), hpTCL("hpTCL",id,this)
  //  : Module("Focusing", id, Source, "Math", "MatlabInterface")
{
    iport1=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(iport1);

    iport2=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(iport2);

    oport1=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(oport1);

    oport2=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(oport2);
}

Focusing::~Focusing(){}

void Focusing::execute()
{

// DECLARATIONS

  //const char *gui; double noise, fcsdg; float tmp1,tmp2;
  double *F,*d,*m,*r; 
  int    Nd,Nm;

  MatrixHandle mh1,mh2,mh3,mh4;
  DenseMatrix  *inp1;   // Sensitivity matrix
  ColumnMatrix *inp2;   // data (right-hand side)
  ColumnMatrix *otp1;   // model
  ColumnMatrix *otp2;   // residual

// OBTAIN SCALAR PARAMETERS FROM GUI
// not working - no gui element
// gui=hpTCL.get().c_str();
//  sscanf(gui,"%g %g",&tmp1,&tmp2);
//  noise=(double)tmp1;
//  fcsdg=(double)tmp2;

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

// ACTUAL OPERATION (EMULATOR FOR NOW)

//fprintf(stderr,"Noise, fcsdg= %g %g\n",noise,fcsdg); 
  for(int i=0;i<Nm;i++) m[i]=F[i];
  for(int i=0;i<Nd;i++) r[i]=d[i];
  
// SEND RESULTS DOWNSTREAM

  oport1->send(mh3);
  oport2->send(mh4);
}

void Focusing::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace MatlabInterface


