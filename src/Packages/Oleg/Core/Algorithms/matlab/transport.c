/* Transport routine, 
   so far brings only double arrays

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
 int     nr,nc; /*number of rows-colums */

 if(mxGetString(prhs[1],hport,sizeof(hport)))
            mexErrMsgTxt("Second argument is not string");


/* 
 printf("%i %i %s\n",wordy,flag,hport);
*/ 

  if(flag==2) /* send operation */
  {
   if(nrhs!=3) mexErrMsgTxt("Incorrect number of input arguments");
   if (mxIsDouble(prhs[2])==0) mexErrMsgTxt("Argument must be a double array.");
   
    db=(double *)mxGetPr(prhs[2]);
    nr=mxGetM(prhs[2]);
    nc=mxGetN(prhs[2]);
    lbuf=nr*nc*sd;
    sprintf(cb,"%i %i %i %i %i\0",lbuf,sd,endian(),nr,nc); 
    bring(wordy,2,hport,lcb,cb);
    if(bring(wordy,flag,hport,lbuf,(char*)db)==NULL)
          mexErrMsgTxt("Not enough memory on receiving side");
  }

  if(flag==1) /* receive operation */
  {
   if(nrhs!=2) mexErrMsgTxt("Incorrect number of receive input arguments");
   if(nlhs!=1) mexErrMsgTxt("Incorrect number of output arguments");
   sscanf(bring(wordy,1,hport,lcb,cb),"%i %i %i %i %i",&lbuf,&sd,&endi,&nr,&nc);
   if(nr*nc*sd!=lbuf) mexErrMsgTxt("Number of bytes does not match");
   if(sd!=8) mexErrMsgTxt("Type is not double");
   else
   {
    plhs[0]=mxCreateDoubleMatrix(nr,nc,mxREAL);
    db=(double *)mxGetPr(plhs[0]);
    if(db==NULL) lbuf=0;
    bring(wordy,1,hport,lbuf,(char*)db);
    if(endi!=endian()) endiswap(lbuf,(char*)db,sd);
   }
  }
 return;
}
