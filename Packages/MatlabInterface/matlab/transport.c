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
*/

/* Transport routine, 
   brings double arrays
          or sparse arrays

   by oleg@cs.utah.edu

   var2=transport([wordy flag],hport,var);

   Based on example from
   http://tcw2.ppsw.rug.nl/documentatie/matlab/techdoc/apiref/mx-c48.html#106922
*/

#include "mex.h"
#include "matrix.h"

char *bring(int wordy,int flag,char *hname,int lbuf,char *buf);
void endiswap(int len,char *buf,int how);
int  endian(void);

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
 double  *pars=(double*)mxGetPr(prhs[0]);
 int     wordy=(int)(pars[0]),
          flag=(int)(pars[1]),
          endi;
 
 char    hport[128],
         cb[128];

 int     lcb=sizeof(cb),
   lbuf,sd=sizeof(double);
 double *db;
 int     nr,nc;       /* Number of rows-colums */
 int     *jc,*ir,nnz; /* Parts for sparse array */
 int     k;

 if(mxGetString(prhs[1],hport,sizeof(hport)))
            mexErrMsgTxt("Second argument is not string");

/* 
 printf("%i %i %s\n",wordy,flag,hport);
*/ 

  if(flag==2) /* SEND OPERATION */
  {
   if(nrhs!=3) mexErrMsgTxt("Incorrect number of input arguments");

   if(mxIsDouble(prhs[2]))  /* SEND DOUBLE ARRAY */
   { 
    db=(double *)mxGetPr(prhs[2]);
    nr=mxGetM(prhs[2]);
    nc=mxGetN(prhs[2]);
    lbuf=nr*nc*sd;
    sprintf(cb,"%i %i %i %i %i\0",lbuf,sd,endian(),nr,nc); 
    bring(wordy,2,hport,lcb,cb);
    if(bring(wordy,flag,hport,lbuf,(char*)db)==NULL)
          mexErrMsgTxt("Not enough memory on receiving side");
    return;
   }
   if(mxIsSparse(prhs[2]))  /* SEND SPARSE ARRAY */
   {
    db=(double *)mxGetPr(prhs[2]);
    nr=mxGetM(prhs[2]);
    nc=mxGetN(prhs[2]);
    ir=(int *)mxGetIr(prhs[2]);
    jc=(int *)mxGetJc(prhs[2]);
    nnz=mxGetNzmax(prhs[2]);
 
    printf("Send sparse nr nc nnz: %i %i %i\n",nr,nc,nnz);
    printf("ir:"); for(k=0;k<nnz;k++) printf(" %i",ir[k]); printf("\n");
    printf("jc:"); for(k=0;k<nc+1;k++) printf(" %i",jc[k]); printf("\n");
    printf("db:"); for(k=0;k<nnz;k++) printf(" %g",db[k]); printf("\n");

    sprintf(cb,"%i %i %i %i %i\0",nnz,9,endian(),nr,nc); 
    bring(wordy,2,hport,lcb,cb);
    if(bring(wordy,flag,hport,nnz*4,(char*)ir)==NULL)
       mexErrMsgTxt("Not enough memory on receiving side");
    bring(wordy,2,(char*)hport,(nc+1)*4,(char*)jc);
    bring(wordy,2,(char*)hport,nnz*8,(char*)db);

    return;
   }
   mexErrMsgTxt("Argument must be a double or sparse array");
  }

  if(flag==1) /* receive operation */
  {
   if(nrhs!=2) mexErrMsgTxt("Incorrect number of receive input arguments");
   if(nlhs!=1) mexErrMsgTxt("Incorrect number of output arguments");
   sscanf(bring(wordy,1,hport,lcb,cb),"%i %i %i %i %i",&lbuf,&sd,&endi,&nr,&nc);
   if(sd!=8) 
   {
     if(sd!=9) mexErrMsgTxt("Type is not double or sparse");
     /* RECEIVE SPARSE */
     nnz=lbuf;

     printf("Receive sparse nr nc nnz: %i %i %i\n",nr,nc,nnz);

     plhs[0]=mxCreateSparse(nr,nc,nnz,mxREAL);
     db=(double *)mxGetPr(plhs[0]);
     ir=(int *)mxGetIr(plhs[0]);
     jc=(int *)mxGetJc(plhs[0]);

     bring(wordy,1,(char*)hport,nnz*4,(char*)ir);
     bring(wordy,1,(char*)hport,(nc+1)*4,(char*)jc);
     bring(wordy,1,(char*)hport,nnz*8,(char*)db);

     if(endi!=endian())
     {
      endiswap(nnz*8,(char*)db,8);
      endiswap((nc+1)*4,(char*)ir,4);
      endiswap(nnz*4,(char*)jc,4);
     }
   }
   else /* RECEIVE DOUBLE */
   {
    if(nr*nc*sd!=lbuf) mexErrMsgTxt("Number of bytes does not match");
    plhs[0]=mxCreateDoubleMatrix(nr,nc,mxREAL);
    db=(double *)mxGetPr(plhs[0]);
    if(db==NULL) lbuf=0;
    bring(wordy,1,hport,lbuf,(char*)db);
    if(endi!=endian()) endiswap(lbuf,(char*)db,sd);
   }
  }
 return;
}
