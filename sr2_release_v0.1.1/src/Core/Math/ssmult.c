/*
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

#ifdef _WIN32
// make sure to get a dllexport defined for these
#include <Core/Math/ssmult.h>
#endif

void ssmult(int beg, int end, int* rows, int* columns,
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

void ssmult_upper(int beg, int end, int* rows, int* columns,
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
void ssmult_uppersub(int nrows, int beg, int end, int* rows, int* columns,
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
