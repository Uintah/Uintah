/****************************TRANSPORT ROUTINE****************************************

  Send character object, open and close client
*************************************************************************************/
#include <Packages/MatlabInterface/Core/Util/transport.h>

namespace MatlabInterface {
using namespace SCIRun;

void transport(int wordy, int flag, const char *hport, char *cmd)
{
  if((flag==4)||(flag==5)) /* OPEN AND CLOSE CLIENT */ 
  {
   bring(wordy-2, flag, hport, 0, NULL);
   return;
  }

  if(flag==2) /* Send character variable */
  {
    int lbuf=strlen(cmd)+1;
    char cb[128];
    int  lcb=sizeof(cb);
    sprintf(cb,"%i %i %i %i %i\n",lbuf,1,endian(),1,lbuf);
    bring(wordy-2, 2, hport, lcb, cb);
    bring(wordy-2, 2, hport, lbuf, cmd);
    return;
  }
  fprintf(stderr,"transport char: wrong flag value %i\n",flag);
}

/****************************TRANSPORT ROUTINE****************************************
  Send and receive MatrixHandle object
*************************************************************************************/

MatrixHandle transport(int wordy, int flag, const char *hport, MatrixHandle mh)
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
  sscanf(bring(wordy, 1, hport, lcb, cb),
	 "%i %i %i %i %i",&lbuf,&sd,&endi,&nr,&nc);

  if(sd==9) 
  {
   if(wordy>0) fprintf(stderr,"Sparse receive operation\n");

   int nnz=lbuf;
   int *rows=scinew int(nc+1);
   int *cols=scinew int(nnz);
   db=scinew double(nnz);

   bring(wordy, 1, hport, nnz*4, (char*)cols);
   bring(wordy, 1, hport, (nc+1)*4, (char*)rows);
   bring(wordy, 1, hport, nnz*8, (char*)db);

   if(endi!=endian())
   {
     endiswap(nnz*8, (char*)db, 8);
     endiswap((nc+1)*4, (char*)rows, 4);
     endiswap(nnz*4, (char*)cols, 4);
   }

   // Transpose internals by creatinc c/r matrix and then transposing.
   SparseRowMatrix *smatrt = scinew SparseRowMatrix(nc,nr,rows,cols,nnz,db);
   smatr = smatrt->transpose();
   delete smatrt;
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
     if(db==NULL) lbuf=0;
     bring(wordy, 1, hport, lbuf, (char*)db);
     if(endi!=endian()) endiswap(lbuf, (char*)db, sd);
    }
    else
    {
     if(wordy>0) fprintf(stderr,"DenseMatrix receive operation\n");
     dmatr=scinew DenseMatrix(nr,nc);
     mh=MatrixHandle(dmatr);
     db=&((*dmatr)[0][0]);
     if(db==NULL) lbuf=0;

     double *tt= scinew double [ nr*nc ];
     if(tt==NULL) lbuf=0;

     bring(wordy, 1, hport, lbuf, (char*)tt);
     if(endi!=endian()) endiswap(lbuf, (char*)tt, sd);

     /* Transposition on receive - Matlab style matrix storage */

     // prt(tt,nr,nc);

     trnsp(tt,db,nr,nc);
     delete [] tt;

     // prt(db,nc,nr);

    }
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
    smatr = smatr->transpose();

    nr=smatr->ncols();
    nc=smatr->nrows();
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
    bring(wordy-2, 2, hport, lcb, cb);

    if(bring(wordy-2, 2, hport, nnz*4, (char*)cols)==NULL)
       fprintf(stderr,"Not enough memory on receiving side");
    bring(wordy-2, 2, hport, (nc+1)*4, (char*)rows);
    bring(wordy-2, 2, hport, nnz*8, (char*)db);
    delete smatr;
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
   bring(wordy-2, 2, hport, lcb, cb);

   /* Transposition on send - Matlab style matrix storage */

   double *tt= scinew double [ nr*nc ];
   trnsp(db,tt,nc,nr);

   if(bring(wordy-2, 2, hport, nr*nc*8, (char*)tt)==NULL)
      fprintf(stderr,"Not enough memory on receiving side");

   delete [] tt;

   return mh;
  }

  fprintf(stderr,"\n transport send ERROR: needs Dense- Column- or SparseRow- Matrix as input\n");
  return mh;
 }
 return mh;
}


/*
   Transposition for matrix storage organization

   p2=trnsp(p1);
   p1(n,m);
   p2(m,n);
*/

void trnsp(double *p1,double *p2,int n,int m)
{
  int n1,m1;

  for(n1=0;n1<n;n1++)
   for(m1=0;m1<m;m1++)
     p2[m1+n1*m]=p1[n1+m1*n]; 

/*  for(n1=0;n1<n*m;n1++) p1[n1]=p2[n1]; */

}




void prt(double *m,int n1,int n2)
{
 int k1,k2;
 for(k1=0;k1<n1;k1++)
 {
  for(k2=0;k2<n2;k2++) printf("%g ",m[k1+n1*k2]);
  printf("\n");
 }
 printf("\n");
}

} // End namespace MatlabInterface
