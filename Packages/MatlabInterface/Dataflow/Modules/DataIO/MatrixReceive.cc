/*

#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#
#  The Original Source Code is SCIRun, released March 12, 2001.
#
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#

 *  MatrixReceive.cc:
 *
 *  Written by:
 *   oleg@cs.utah.edu
 *   01Mar14
 *
 */

#include <stdio.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Packages/MatlabInterface/share/share.h>

namespace MatlabInterface
{
      char *bring(int wordy,int flag,char *hostport,int lbuf,char *buf);
      void endiswap(int lbuf, char *buf,int num);
      int  endian(void);

using namespace SCIRun;

class MatlabInterfaceSHARE MatrixReceive : public Module 
{
 GuiString hpTCL;
 MatrixIPort *imat1;
 MatrixOPort *omat1;
 MatrixOPort *omat2;

public:
  MatrixReceive(const string& id);
  virtual ~MatrixReceive();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MatlabInterfaceSHARE Module* make_MatrixReceive(const string& id) {
  return scinew MatrixReceive(id);
}

MatrixReceive::MatrixReceive(const string& id)
  : Module("MatrixReceive", id, Source, "DataIO", "MatlabInterface") , hpTCL("hpTCL",id,this)
{
}

MatrixReceive::~MatrixReceive(){ }

void MatrixReceive::execute()
{
  imat1 = (MatrixIPort *)get_iport("host:port string");
  omat1 = (MatrixOPort *)get_oport("host:port string");
  omat2 = (MatrixOPort *)get_oport("Received Matrix");

  if (!imat1) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!omat1) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!omat2) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
/* OBTAIN HOST:PORT INFORMATION */

 const char *hport=hpTCL.get().c_str();
 DenseMatrix     *matr;
 MatrixHandle    mIH;
 SparseRowMatrix *smatr;

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
   fprintf(stderr,"MatrixSend needs DenseMatrix for host:port input\n");
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
  ColumnMatrix *ccc;


  sscanf(bring(wordy,1,(char*)hport,lcb,cb),"%i %i %i %i %i",&lbuf,&sd,&endi,&nr,&nc);
  if(sd!=8) 
  {
    if(sd!=9)  fprintf(stderr,"Sending type is not double and not sparse");

    /* SPARSE MATRIX RECEIVE OPERATION */

    int nnz=lbuf;
    int *rows=scinew int(nc+1);
    int *cols=scinew int(nnz);
    double *d=scinew double(nnz);

    bring(wordy,1,(char*)hport,nnz*4,(char*)cols);
    bring(wordy,1,(char*)hport,(nc+1)*4,(char*)rows);
    bring(wordy,1,(char*)hport,nnz*8,(char*)d);

    if(endi!=endian()) 
    {
      endiswap(nnz*8,(char*)d,8);
      endiswap((nc+1)*4,(char*)rows,4);
      endiswap(nnz*4,(char*)cols,4);
    }

    smatr=scinew SparseRowMatrix(nr,nc,rows,cols,nnz,d);
    mm=MatrixHandle(smatr);

  }
  else
  {

    /* DENSE AND COLUMNMATRIX RECEIVE OPERATION */

    double *db;
    if(nc==1)
    {
     ccc=scinew ColumnMatrix(nr);
     mm=MatrixHandle(ccc);
     db=&((*ccc)[0]);
    }
    else
    {
     ttt=scinew DenseMatrix(nr,nc);
     mm=MatrixHandle(ttt);
     db=&((*ttt)[0][0]);
    }

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

} // End namespace MatlabInterface
