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

 *  MatrixSend.cc:
 *
 *  Written by:
 *   oleg@cs.utah.edu
 *   01Mar14 
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Packages/MatlabInterface/share/share.h>
#include <Packages/MatlabInterface/Core/Util/bring.h>
#include <stdio.h>

namespace MatlabInterface
{

using namespace SCIRun;

class MatlabInterfaceSHARE MatrixSend : public Module 
{
 GuiString hpTCL;
 MatrixIPort *imat1;
 MatrixIPort *imat2;
 MatrixOPort *omat;

 public:
  MatrixSend(const string& id);
  virtual ~MatrixSend();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MatlabInterfaceSHARE Module* make_MatrixSend(const string& id) 
{
  return scinew MatrixSend(id);
}

MatrixSend::MatrixSend(const string& id)
: Module("MatrixSend", id, Filter, "DataIO", "MatlabInterface"), hpTCL("hpTCL",id,this)
//  : Module("MatrixSend", id, Source, "DataIO", "MatlabInterface")
{
}

MatrixSend::~MatrixSend(){}

void MatrixSend::execute()
{

  imat1 = (MatrixIPort *)get_iport("host:port string");
  imat2 = (MatrixIPort *)get_iport("Matrix being sent");
  omat = (MatrixOPort *)get_oport("host:port string");

  if (!imat1) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!imat2) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!omat) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  /* OBTAIN HOST:PORT INFORMATION */

 hpTCL.reset();
 // cerr << "hpTCL : " << hpTCL.get() << endl; 
 string ss=hpTCL.get();
 const char *hport=ss.c_str();
 DenseMatrix *matr;
 ColumnMatrix *cmatr;
 SparseRowMatrix *smatr;
 MatrixHandle mIH;
 double *db;
 char cb[128];
 int  lcb=sizeof(cb);
 int  wordy=0;

 if(strcmp(hport,"")!=0)  
 {
   if(wordy>1) fprintf(stderr,"obtain host:port from gui\n");
   matr=scinew DenseMatrix(10,1);
   strcpy((char*)&((*matr)[0][0]),hport); 
   mIH=MatrixHandle(matr);
 }
 else
 {
  if(wordy>1) fprintf(stderr,"obtain host:port from input port\n");
  imat1->get(mIH);
  matr=dynamic_cast<DenseMatrix*>(mIH.get_rep()); //upcast
  if(matr==NULL)
  {
    fprintf(stderr,"MatrixSend needs DenseMatrix for host:port input\n");
   return;
  }
  hport=(char*)&((*matr)[0][0]);
 }
 if(wordy>0) fprintf(stderr,"host:port is %s\n",hport);

/* ACTUAL SEND OPERATION */

 MatrixHandle mh;
 imat2->get(mh);
 DenseMatrix *tmp=dynamic_cast<DenseMatrix*>(mh.get_rep()); 

 if(tmp==NULL)
 {
   cmatr=dynamic_cast<ColumnMatrix*>(mh.get_rep()); //upcast
   if(cmatr==NULL)
   {
    smatr=dynamic_cast<SparseRowMatrix*>(mh.get_rep());
    if(smatr==NULL)
    {
     fprintf(stderr,"MatrixSend needs Dense-  Column- or SparseRow- Matrix as input\n");
     return;
    }
    /* SEND SPARSE MATRIX */
    if(wordy>0) fprintf(stderr,"send sparse matrix\n");

    int nr=smatr->nrows();
    int nc=smatr->ncols();
    int *rows=smatr->get_row();
    int *cols=smatr->get_col();
    double *d=smatr->get_val();
    int nnz=smatr->get_nnz();


    if(wordy>1) 
    {
     int k;
     printf("Send sparse nr nc nnz: %i %i %i\n",nr,nc,nnz);
     printf("rows:"); for(k=0;k<nc+1;k++) printf(" %i",rows[k]); printf("\n");
     printf("cols:"); for(k=0;k<nnz;k++) printf(" %i",cols[k]); printf("\n");
     printf("d   :"); for(k=0;k<nnz;k++) printf(" %g",d[k]); printf("\n");
    } 

    sprintf(cb,"%i %i %i %i %i\n",nnz,9,endian(),nr,nc);
    bring(wordy-2,2,(char *)hport,lcb,cb);

    if(bring(wordy-2,2,(char*)hport,nnz*4,(char*)cols)==NULL)
       fprintf(stderr,"Not enough memory on receiving side");
    bring(wordy-2,2,(char*)hport,(nc+1)*4,(char*)rows);
    bring(wordy-2,2,(char*)hport,nnz*8,(char*)d);

    omat->send(mIH);
    return;
   }
   db=&((*cmatr)[0]);
 }
 else db=&((*tmp)[0][0]);

 int nr=mh->nrows();
 int nc=mh->ncols();
 int  sd=8;
 int  lbuf=nc*nr*sd;

 if(wordy>0) fprintf(stderr,"send double data\n");
 if(wordy>1) for(int i=0;i<nr*nc;i++) fprintf(stderr,"%g ",db[i]);

 sprintf(cb,"%i %i %i %i %i\n",lbuf,sd,endian(),nr,nc);
 if(wordy>1) fprintf(stderr,"sending buffer: %i %s\n",lcb,cb);
 bring(wordy-2,2,(char *)hport,lcb,cb);

 if(bring(wordy-2,2,(char*)hport,lbuf,(char*)db)==NULL)
      fprintf(stderr,"Not enough memory on receiving side");


/* SEND HOST:PORT DOWNSTREAM */

 omat->send(mIH);

}

void MatrixSend::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace MatlabInterface


