/****************************TRANSPORT ROUTINE****************************************

  Send character object, open and close client
*************************************************************************************/
#include <Packages/MatlabInterface/Core/Util/transport.h>

namespace MatlabInterface {
using namespace SCIRun;

void transport(int wordy,int flag,char *hport,char *cmd)
{
  if((flag==4)||(flag==5)) /* OPEN AND CLOSE CLIENT */ 
  {
   bring(wordy-2,flag,(char*)hport,0,NULL);
   return;
  }

  if(flag==2) /* Send character variable */
  {
    int lbuf=strlen(cmd)+1;
    char cb[128];
    int  lcb=sizeof(cb);
    sprintf(cb,"%i %i %i %i %i\n",lbuf,1,endian(),1,lbuf);
    bring(wordy-2,2,(char*)hport,lcb,cb);
    bring(wordy-2,2,(char*)hport,lbuf,cmd);
    return;
  }
  fprintf(stderr,"transport char: wrong flag value %i\n",flag);
}

/****************************TRANSPORT ROUTINE****************************************
  Send and receive MatrixHandle object
*************************************************************************************/

MatrixHandle transport(int wordy,int flag,char *hport,MatrixHandle mh)
{ 
 char cb[128];
 int  lcb=sizeof(cb);
 DenseMatrix     *dmatr;
 ColumnMatrix    *cmatr;
 SparseRowMatrix *smatr;
 double *db;
 int    nr,nc;

 if(flag==1) /* RECEIVE OPERATION */
 {
  int    endi,lbuf,sd;
  sscanf(bring(wordy,1,(char*)hport,lcb,cb),"%i %i %i %i %i",&lbuf,&sd,&endi,&nr,&nc);

  if(sd==9) 
  {
   if(wordy>0) fprintf(stderr,"Sparse receive operation\n");

   int nnz=lbuf;
   int *rows=scinew int(nc+1);
   int *cols=scinew int(nnz);
   db=scinew double(nnz);

   bring(wordy,1,(char*)hport,nnz*4,(char*)cols);
   bring(wordy,1,(char*)hport,(nc+1)*4,(char*)rows);
   bring(wordy,1,(char*)hport,nnz*8,(char*)db);

   if(endi!=endian())
   {
     endiswap(nnz*8,(char*)db,8);
     endiswap((nc+1)*4,(char*)rows,4);
     endiswap(nnz*4,(char*)cols,4);
   }

   smatr=scinew SparseRowMatrix(nr,nc,rows,cols,nnz,db);
   mh=MatrixHandle(smatr);
   return(mh);
  }

  if(sd==8) 
  {
    if(nc==1)
    {
     if(wordy>0) fprintf(stderr,"ColumnMatrix receive operation\n");
     cmatr=scinew ColumnMatrix(nr);
     mh=MatrixHandle(cmatr);
     db=&((*cmatr)[0]);
    }
    else
    {
     if(wordy>0) fprintf(stderr,"DenseMatrix receive operation\n");
     dmatr=scinew DenseMatrix(nr,nc);
     mh=MatrixHandle(dmatr);
     db=&((*dmatr)[0][0]);
    }

    if(db==NULL) lbuf=0;

    bring(wordy,1,(char*)hport,lbuf,(char*)db);
    if(endi!=endian()) endiswap(lbuf,(char*)db,sd);
    return(mh);
  }

  fprintf(stderr,"\n transport rcv ERROR: type is not double and not sparse\n");
  return(mh);
 }

 if(flag==2) /* SEND OPERATION */
 {
  dmatr=dynamic_cast<DenseMatrix*>(mh.get_rep());
  cmatr=dynamic_cast<ColumnMatrix*>(mh.get_rep()); 
  smatr=dynamic_cast<SparseRowMatrix*>(mh.get_rep());

  if(smatr!=NULL) 
  {
    if(wordy>0) fprintf(stderr,"Send sparse row matrix\n");

    nr=smatr->nrows();
    nc=smatr->ncols();
    int *rows=smatr->get_row();
    int *cols=smatr->get_col();
    db=smatr->get_val();
    int nnz=smatr->get_nnz();

    if(wordy>1)
    {
     int k;
     printf("Send sparse nr nc nnz: %i %i %i\n",nr,nc,nnz);
     printf("rows:"); for(k=0;k<nc+1;k++) printf(" %i",rows[k]); printf("\n");
     printf("cols:"); for(k=0;k<nnz;k++) printf(" %i",cols[k]); printf("\n");
     printf("db  :"); for(k=0;k<nnz;k++) printf(" %g",db[k]); printf("\n");
    }

    sprintf(cb,"%i %i %i %i %i\n",nnz,9,endian(),nr,nc);
    bring(wordy-2,2,(char *)hport,lcb,cb);

    if(bring(wordy-2,2,(char*)hport,nnz*4,(char*)cols)==NULL)
       fprintf(stderr,"Not enough memory on receiving side");
    bring(wordy-2,2,(char*)hport,(nc+1)*4,(char*)rows);
    bring(wordy-2,2,(char*)hport,nnz*8,(char*)db);
    return mh;
  }

  db=NULL;
  if(cmatr!=NULL) 
  {
   if(wordy>0) fprintf(stderr,"Send ColumnMatrix\n");
   db=&((*cmatr)[0]);
  }
  if(dmatr!=NULL) 
  {
   if(wordy>0) fprintf(stderr,"Send DoubleMatrix\n");
   db=&((*dmatr)[0][0]);
  }
  if(db!=NULL)  /* SEND DOUBLE DATA */
  {
   nr=mh->nrows();
   nc=mh->ncols();

   if(wordy>0) fprintf(stderr,"send double data\n");
   if(wordy>1) for(int i=0;i<nr*nc;i++) fprintf(stderr,"%g ",db[i]);

   sprintf(cb,"%i %i %i %i %i\n",nr*nc*8,8,endian(),nr,nc);
   if(wordy>1) fprintf(stderr,"sending buffer: %i %s\n",lcb,cb);
   bring(wordy-2,2,(char *)hport,lcb,cb);

   if(bring(wordy-2,2,(char*)hport,nr*nc*8,(char*)db)==NULL)
      fprintf(stderr,"Not enough memory on receiving side");

   return mh;
  }

  fprintf(stderr,"\n transport send ERROR: needs Dense- Column- or SparseRow- Matrix as input\n");
  return mh;
 }
 return mh;
}

} // End namespace MatlabInterface

