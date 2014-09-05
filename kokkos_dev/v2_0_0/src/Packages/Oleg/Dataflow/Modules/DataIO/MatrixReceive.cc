/*
 *  MatrixReceive.cc:
 *
 *  Written by:
 *   oleg
 *   TODAY'S DATE HERE
 *
 */

#include <stdio.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Packages/Oleg/share/share.h>

namespace Oleg
{
      char *bring(int wordy,int flag,char *hostport,int lbuf,char *buf);
      void endiswap(int lbuf, char *buf,int num);
      int  endian(void);

using namespace SCIRun;

class OlegSHARE MatrixReceive : public Module 
{
 GuiString hpTCL;
 MatrixIPort *imat1;
 MatrixOPort *omat1;
 MatrixOPort *omat2;

public:
  MatrixReceive(const clString& id);
  virtual ~MatrixReceive();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" OlegSHARE Module* make_MatrixReceive(const clString& id) {
  return scinew MatrixReceive(id);
}

MatrixReceive::MatrixReceive(const clString& id)
  : Module("MatrixReceive", id, Source, "DataIO", "Oleg") , hpTCL("hpTCL",id,this)
{
    imat1=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(imat1);

    omat1=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(omat1);

    omat2=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(omat2);

}

MatrixReceive::~MatrixReceive(){ }

void MatrixReceive::execute()
{
/* OBTAIN HOST:PORT INFORMATION */

 const char *hport=hpTCL.get()();
 DenseMatrix *matr;
 MatrixHandle mIH;

 if(strcmp(hport,"")!=0)
 {
   fprintf(stderr,"Seed matr and send downstream\n");
   matr=scinew DenseMatrix(10,1);
   strcpy((char*)&((*matr)[0][0]),hport);
   mIH=MatrixHandle(matr);
 }

 if(strcmp(hport,"")==0)
 {
  imat1->get(mIH);
  matr=dynamic_cast<DenseMatrix*>(mIH.get_rep()); //upcast
  if(matr==NULL)
  {
   fprintf(stderr,"MatrixSend needs DenseMatrix as input\n");
   return;
  }
  hport=(char*)&((*matr)[0][0]);
  fprintf(stderr,"Name: %s\n",hport);
 }

/* ACTUAL RECEIVE OPERATION */

  int nr,nc,lbuf,sd;
  DenseMatrix *ttt;
  MatrixHandle mm;
  char cb[128];
  int  lcb=sizeof(cb);
  int  endi,wordy=2;

  sscanf(bring(wordy,1,(char*)hport,lcb,cb),"%i %i %i %i %i",&lbuf,&sd,&endi,&nr,&nc);
  if(sd!=8) fprintf(stderr,"Sending type is not double");
  else
  {
    ttt=scinew DenseMatrix(nr,nc);
    mm=MatrixHandle(ttt);
    double *db=&((*ttt)[0][0]);
    if(db==NULL) lbuf=0;
    bring(wordy,1,(char*)hport,lbuf,(char*)db);
    if(endi!=endian()) endiswap(lbuf,(char*)db,sd);
  }
  omat2->send(mm);

// fprintf(stderr,"Receive double data\n");
// for(int i=0;i<nr*nc;i++) fprintf(stderr,"%g ",db[i]);

/* SEND HOST:PORT DOWNSTREAM */

  omat1->send(mIH);
}

void MatrixReceive::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Oleg
