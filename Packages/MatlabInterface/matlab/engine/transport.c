/*
#
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

*/

/* Transport routine, 
   for       double arrays
          or sparse arrays
          or character arrays

   by oleg@cs.utah.edu

   var2=transport(wordy,flag,hport,var);

   Based on example from
   http://tcw2.ppsw.rug.nl/documentatie/matlab/techdoc/apiref/mx-c48.html#106922
   http://www.mathworks.co.uk/access/helpdesk_r11/help/techdoc/apiref/mxisdouble.shtml?cmdname=MxIsDouble
   examples/refbook/revord.c - matlab examples
*/

/* #include "mex.h" */
#include "matrix.h"

char *bring(int wordy,int flag,char *hname,int lbuf,char *buf);
void endiswap(int len,char *buf,int how);
int  endian(void);

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
 double  *pars1=(double*)mxGetPr(prhs[0]);
 int     wordy=(int)(pars1[0]);
 double  *pars2=(double*)mxGetPr(prhs[1]);
 int     flag=(int)(pars2[0]),
         endi;
 
 char    hport[128],
         cb[128],
         *cbuf;

 int     lcb=sizeof(cb),
         lbuf,sd=sizeof(double);
 double *db;
 int     nr,nc;       /* Number of rows-colums */
 int     *jc,*ir,nnz; /* Parts for sparse array */

  if(flag==5) /* CLOSE */
  {
    bring(wordy-2,5,NULL,0,NULL);
    return;
  }

  if(mxGetString(prhs[2],hport,sizeof(hport)))
            mexErrMsgTxt("Second argument is not string");

  if(flag==3) /* SERVER OPEN  */
    bring(wordy-2,3,hport,0,NULL);

  if(flag==4) /* CLIENT OPEN  */
    bring(wordy-2,4,hport,0,NULL);


  if(flag==2) /* SEND OPERATION */
  {
   if(nrhs!=4) mexErrMsgTxt("Incorrect number of input arguments");

   if(mxIsDouble(prhs[3]))  /* SEND DOUBLE ARRAY */
   { 
    db=(double *)mxGetPr(prhs[3]);
    nr=mxGetM(prhs[3]);
    nc=mxGetN(prhs[3]);
    lbuf=nr*nc*sd;
    sprintf(cb,"%i %i %i %i %i\0",lbuf,sd,endian(),nr,nc); 
    bring(wordy-2,2,hport,lcb,cb);
    if(bring(wordy-2,2,hport,lbuf,(char*)db)==NULL)
          mexErrMsgTxt("Not enough memory on receiving side");
    return;
   }

   if(mxIsSparse(prhs[3]))  /* SEND SPARSE ARRAY */
   {
    db=(double *)mxGetPr(prhs[3]);
    nr=mxGetM(prhs[3]);
    nc=mxGetN(prhs[3]);
    ir=(int *)mxGetIr(prhs[3]);
    jc=(int *)mxGetJc(prhs[3]);
    nnz=mxGetNzmax(prhs[3]);
 
/*
    printf("Send sparse nr nc nnz: %i %i %i\n",nr,nc,nnz);
    printf("ir:"); for(k=0;k<nnz;k++) printf(" %i",ir[k]); printf("\n");
    printf("jc:"); for(k=0;k<nc+1;k++) printf(" %i",jc[k]); printf("\n");
    printf("db:"); for(k=0;k<nnz;k++) printf(" %g",db[k]); printf("\n");
*/

    sprintf(cb,"%i %i %i %i %i\0",nnz,9,endian(),nr,nc); 
    bring(wordy-2,2,hport,lcb,cb);
    if(bring(wordy-2,flag,hport,nnz*4,(char*)ir)==NULL)
       mexErrMsgTxt("Not enough memory on receiving side");
    bring(wordy-2,2,(char*)hport,(nc+1)*4,(char*)jc);
    bring(wordy-2,2,(char*)hport,nnz*8,(char*)db);

    return;
   }

   if(mxIsChar(prhs[3]))  /* SEND CHARACTER STRING */
   {
     nr=mxGetM(prhs[3]);
     if(nr!=1) mexErrMsgTxt("Only row char vectors can be send");
     nc=mxGetN(prhs[3]);
     lbuf=nr*nc+1;
     cbuf=mxCalloc(lbuf, sizeof(char));
     if(cbuf==NULL)
      mexErrMsgTxt("Cannot allocate memory for string");
     if(mxGetString(prhs[3], cbuf, lbuf))
      mexErrMsgTxt("Not enough space to copy the string");

     sprintf(cb,"%i %i %i %i %i\0",lbuf,1,endian(),nr,nc);
     bring(wordy-2,2,hport,lcb,cb);
     if(bring(wordy-2,2,hport,lbuf,cbuf)==NULL)
          mexErrMsgTxt("Not enough memory on receiving side");
     mxFree((void*)cbuf); 
     return;
   }

   mexErrMsgTxt("Send argument must be double, sparse or character");
  }

  if(flag==1) /* RECEIVE OPERATION */
  {

   if(nrhs!=3) mexErrMsgTxt("Incorrect number of receive input arguments");
   if(nlhs!=1) mexErrMsgTxt("Incorrect number of output arguments");
   sscanf(bring(wordy-2,1,hport,lcb,cb),"%i %i %i %i %i",&lbuf,&sd,&endi,&nr,&nc);

   if(sd==9) /* RECEIVE SPARSE */
   {
     nnz=lbuf;

   /*
     printf("Receive sparse nr nc nnz: %i %i %i\n",nr,nc,nnz);
    */

     plhs[0]=mxCreateSparse(nr,nc,nnz,mxREAL);
     db=(double *)mxGetPr(plhs[0]);
     ir=(int *)mxGetIr(plhs[0]);
     jc=(int *)mxGetJc(plhs[0]);

     bring(wordy-2,1,(char*)hport,nnz*4,(char*)ir);
     bring(wordy-2,1,(char*)hport,(nc+1)*4,(char*)jc);
     bring(wordy-2,1,(char*)hport,nnz*8,(char*)db);

     if(endi!=endian())
     {
      endiswap(nnz*8,(char*)db,8);
      endiswap((nc+1)*4,(char*)ir,4);
      endiswap(nnz*4,(char*)jc,4);
     }
     return;
   }

   if(sd==8) /* RECEIVE DOUBLE */
   {
    if(nr*nc*sd!=lbuf) mexErrMsgTxt("Number of bytes does not match");
    plhs[0]=mxCreateDoubleMatrix(nr,nc,mxREAL);
    db=(double *)mxGetPr(plhs[0]);
    if(db==NULL) lbuf=0;
    bring(wordy-2,1,hport,lbuf,(char*)db);
    if(endi!=endian()) endiswap(lbuf,(char*)db,sd);
    return;
   }

   if(sd==1) /* RECEIVE CHARACTER STRING */
   {
    if(nr!=1) mexErrMsgTxt("Only row char vectors can be received");
    cbuf=mxCalloc(lbuf, sizeof(char));
    if(cbuf==NULL) lbuf=0;
    bring(wordy-2,1,hport,lbuf,cbuf);
    if(cbuf[lbuf-1]!='\0') mexErrMsgTxt("No termination character in received string");
    plhs[0] = mxCreateString(cbuf);
    mxFree((void*)cbuf);
    return;
   }

   mexErrMsgTxt("Receive type is not double or sparse");
  }

}
