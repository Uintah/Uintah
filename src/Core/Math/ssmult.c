/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <Core/share/share.h>

SCICORESHARE void ssmult(int beg, int end, int* rows, int* columns,
	    double* a, double* xp, double* bp)
{
    int i, j;
    for(i=beg;i<end;i++){
	register double sum=0;
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	for(j=row_idx;j<next_idx;j++){
	    sum+=a[j]*xp[columns[j]];
	}
	bp[i]=sum;
   
    }
}

SCICORESHARE void ssmult_upper(int beg, int end, int* rows, int* columns,
		  double* a, double* xp, double* bp)
{
    int i, j;
    for(i=beg;i<end;i++){
	bp[i]=0;
    }
    for(i=beg;i<end;i++){
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	register double r=xp[i];
	register double sum=bp[i]+a[row_idx]*xp[columns[row_idx]];
	for(j=row_idx+1;j<next_idx;j++){
	    register double aj=a[j];
	    register int cj=columns[j];
	    sum+=aj*xp[cj];
	    bp[cj]+=aj*r;
	}
	bp[i]=sum;
    }
}


/* This is too slow....*/
SCICORESHARE void ssmult_uppersub(int nrows, int beg, int end, int* rows, int* columns,
		     double* a, double* xp, double* bp)
{
    int i, j;
    for(i=beg;i<end;i++){
	bp[i]=0;
    }
    for(i=0;i<beg;i++){
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	/*int j=row_idx+1; */
	/*while(j<next_idx && columns[j]<beg)j++; */
	int l=row_idx+1;
	int h=next_idx;
	int j;
	while(h-l>1){
	    int m=(l+h)/2;
	    if(beg<columns[m]){
		h=m;
	    } else if(beg>columns[m]){
		l=m;
	    } else {
		l=m;
		break;
	    }
	}
	/*if(!(columns[l]<=beg && columns[h]>=beg) && l<next_idx)
	   printf("%d %d %d, %d l=%d h=%d %d\n", beg, columns[l], columns[h],row_idx+1,l,h,next_idx);*/
	if(columns[l]<beg)
	    l++;
	j=l;
	while(j<next_idx && columns[j]<end){
	    bp[columns[j]]+=a[j]*xp[i];
	    j++;
	}
    }
    for(;i<end;i++){
	int l, h;
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	/*register double r=xp[i];*/
	register double sum=bp[i]+a[row_idx]*xp[columns[row_idx]];
	for(j=row_idx+1;j<next_idx;j++){
	    register double aj=a[j];
	    register int cj=columns[j];
	    sum+=aj*xp[cj];
	}
	bp[i]=sum;
	l=row_idx+1;
	h=next_idx;
	while(h-l>1){
	    int m=(l+h)/2;
	    if(beg<columns[m]){
		h=m;
	    } else if(beg>columns[m]){
		l=m;
	    } else {
		l=m;
		break;
	    }
	}
	if(columns[l]<beg)
	    l++;
	j=l;

	/*j=row_idx+1;*/
	/*while(j<next_idx && columns[j]<beg)j++; */
	while(j<next_idx && columns[j]<end){
	    bp[columns[j]]+=a[j]*xp[i];
	    j++;
	}
    }
    for(;i<nrows;i++){
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	int l=row_idx+1;
	int h=next_idx;
	int j;
	while(h-l>1){
	    int m=(l+h)/2;
	    if(beg<columns[m]){
		h=m;
	    } else if(beg>columns[m]){
		l=m;
	    } else {
		l=m;
		break;
	    }
	}
	if(columns[l]<beg)
	    l++;
	j=l;
	/*int j=row_idx+1; */
	/*while(j<next_idx && columns[j]<beg)j++; */
	while(j<next_idx && columns[j]<end){
	    bp[columns[j]]+=a[j]*xp[i];
	    j++;
	}
    }
}
