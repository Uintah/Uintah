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

/*
 *  ExtraMath.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */
 
#include <Packages/CardioWave/Core/Algorithms/ExtraMath.h>
 
namespace SCIRun {
 
ExtraMath::ExtraMath()
{
}
 
ExtraMath::~ExtraMath()
{
}

int ExtraMath::computebandwidth(MatrixHandle matH)
{
  if (matH->is_dense())
  {
    DenseMatrix* d = matH->as_dense();
    int n,m;
    n = d->nrows();
    m = d->ncols();
    double *ptr = d->get_data_pointer();
    
    int bw = 0;
    int q;
    int strt,fnsh;
    for (int p=0;p<n;p++)
    {
      strt = 0;
      fnsh = 0;
      for(q=0;q<m;q++)
      {
        if (ptr[q] != 0.0) break;
      }
      strt = q;
      fnsh = q+1;
      for(;q<m;q++)
      {
        if (ptr[q] != 0.0) fnsh = q+1;
      }
      
      if ((fnsh-strt) > bw) bw = fnsh-strt;
    }
    return(bw);
  }

  if (matH->is_column())
  {
    ColumnMatrix* c = matH->as_column();
    int n;
    n = c->nrows();
    double *ptr = c->get_data_pointer();
    for (int p=0;p<n;p++) if (ptr[p] != 0.0) return(1);
    return(0);
  }
  
  if (matH->is_sparse())
  {
    SparseRowMatrix* s = matH->as_sparse();
    int n,m;
    n = s->nrows();
    m = s->ncols();
    int *rr = s->rows;
    int *cc = s->columns;

    int bw = 0;
    int q;
    int strt,fnsh;

    for (int p=0;p<n;p++)
    {
      strt = 0;
      fnsh = 0;
      if (rr[p+1] > rr[p])
      {
        strt = cc[rr[p]];
        fnsh = cc[rr[p]];
        bw = 1;
      }
      else
      {
        continue;
      }
      for (int r = (rr[p]+1);r<rr[p+1];r++)
      {
          if (cc[r] < strt) strt = cc[r];
          if (cc[r] > fnsh) fnsh = cc[r];
      }
      if ((fnsh-strt+1) > bw) bw = (fnsh-strt)+1;
    }
    return(bw);
  }
  return(0);
}
 
 
void ExtraMath::rcm(MatrixHandle im,MatrixHandle& om,MatrixHandle& mapping,bool calcmapping,MatrixHandle& imapping,bool calcimapping)
{
  int *rr, *cc;
  double *d;
  int n,m,nnz;
  
  if (im->is_sparse() == false) return;
  SparseRowMatrix* sim = im->as_sparse();
  
  om = sim->clone();
  m  = sim->nrows();
  n  = sim->ncols();
  nnz = sim->nnz;
  rr = sim->rows;
  cc = sim->columns;
  d  = sim->a;
  
  // reserve mapping space
  
  int *mrr, *mcc;
  double *md;
  int *imrr, *imcc;
  double *imd;

  int *drr, *dcc;
  double *dd;

  if (calcmapping)
  {
    mrr = scinew int[m+1];
    mcc = scinew int[m];
    md  = scinew double[m];
    mapping = scinew SparseRowMatrix(m,m,mrr,mcc,m,md);
  }

  if (calcimapping)
  {
    imrr = scinew int[m+1];
    imcc = scinew int[m];
    imd  = scinew double[m];
    imapping = scinew SparseRowMatrix(m,m,imrr,imcc,m,imd);
  }

  SparseRowMatrix* som = om->as_sparse();  
  drr = som->rows;
  dcc = som->columns;
  dd  = som->a;

  int *Q, *R, *S, *X;
  int *degree;
  int ns,nr,nq,nx,nss;
  
  // Count number of connections
  degree = drr;
  for (int p = 0;p<m;p++) degree[p] = (degree[p+1] - degree[p]);
  
  // clear the mask of already processed nodes

  X = scinew int[m];
  Q = scinew int[m];
  S = scinew int[m];
  R = scinew int[m];
  
  for (int p=0;p<m;p++) Q[p] = 0;
  
  int root = 0;
  for (int p=0;p<m;p++) if((degree[p] < degree[root])&&(Q[p] == 0)) root = p;

  nss = 0;
  nr = 0;
  ns = 0;
  nq = 0;
  nx = 0;
  
  R[nr++] = root;
  Q[root] = 1; nq++;
  X[nx++] = root;
  
  nr = 0;
  for (int p = rr[root];p<rr[root+1];p++) if (Q[cc[p]] == 0) { R[nr++] = cc[p]; Q[cc[p]] = 1; nq++; }

 

 
  while(1)
  {
    if(nr > 0) 
    { 
      for (int i =0;i<nr;i++)
      {
        int t;
        int k = i;
        // sort entries by order
        for (int j=i+1;j<nr;j++) { if (degree[R[k]] > degree[R[j]]) { k = j;}
        if (k > i)  {t = R[i]; R[i] = R[k]; R[k] = t;}  }
      }
      for (int j=0;j<nr;j++) { X[nx++] = R[j]; S[ns++] = R[j]; }
      nr  = 0;
    }
    if (nx == m) break;
    
    if (nss < ns)
    {
      root = S[nss++];
      nr = 0;
      for (int p = rr[root];p<rr[root+1];p++) if (Q[cc[p]] == 0) { R[nr++] = cc[p]; Q[cc[p]] = 1; nq++; }
    }
    else
    {
      for (int p=0;p<m;p++) if(Q[p] == 0) { root = p; break;}
      for (int p=0;p<m;p++) if((degree[p] < degree[root])&&(Q[p] == 0)) root = p;
      Q[root] = 1; nq++;
      X[nx++] = root;
      nr = 0;
      for (int p = rr[root];p<rr[root+1];p++) if (Q[cc[p]] == 0) { R[nr++] = cc[p]; Q[cc[p]] = 1; nq++; }
    }    
  }

  int t;
  // do reverse ordering CM -> RCM
  for (int p=0;p<m;p++) { t = X[p]; X[p] = X[m-1-p]; X[m-1-p] = t; }
  
  // finish mapping matrix and inverse mapping matrix
  if (calcmapping)
  {
    for (int p=0;p<(m+1);p++) { mrr[p] = p; }
    for (int p=0;p<m;p++) { md[p] = 1.0;  }
    for (int p=0;p<m;p++) { mcc[p] = X[p]; }
  }
  
  if (calcimapping)
  {
    for (int p=0;p<(m+1);p++) { imrr[p] = p;}
    for (int p=0;p<m;p++) { imd[p] = 1.0; }
    for (int p=0;p<m;p++) { imcc[X[p]] = p; }
  }
  
   for (int p=0;p<m;p++) { Q[X[p]] = p; }
  
  // finish the reordering new matrix
  int i,j;
  drr[0] = 0;
  i = 0;
  for (int p=0;p<m;p++)
  {
    j = Q[p];
    drr[p+1] = drr[p]+(rr[j+1]-rr[j]);
    for (int r=rr[j];r<rr[j+1];r++)
    {
      dcc[i] = X[cc[r]];
      dd[i] = d[r];
      i++;
    }
  }

  delete[] X;
  delete[] Q;
  delete[] S;
  delete[] R;


} 
 

void ExtraMath::cm(MatrixHandle im,MatrixHandle& om,MatrixHandle& mapping,bool calcmapping,MatrixHandle& imapping,bool calcimapping)
{
 int *rr, *cc;
  double *d;
  int n,m,nnz;
  
  if (im->is_sparse() == false) return;
  SparseRowMatrix* sim = im->as_sparse();
  
  om = sim->clone();
  m  = sim->nrows();
  n  = sim->ncols();
  nnz = sim->nnz;
  rr = sim->rows;
  cc = sim->columns;
  d  = sim->a;
  
  // reserve mapping space
  
  int *mrr, *mcc;
  double *md;
  int *imrr, *imcc;
  double *imd;

  int *drr, *dcc;
  double *dd;

  if (calcmapping)
  {
    mrr = scinew int[m+1];
    mcc = scinew int[m];
    md  = scinew double[m];
    mapping = scinew SparseRowMatrix(m,m,mrr,mcc,m,md);
  }

  if (calcimapping)
  {
    imrr = scinew int[m+1];
    imcc = scinew int[m];
    imd  = scinew double[m];
    imapping = scinew SparseRowMatrix(m,m,imrr,imcc,m,imd);
  }

  SparseRowMatrix* som = om->as_sparse();  
  drr = som->rows;
  dcc = som->columns;
  dd  = som->a;

  int *Q, *R, *S, *X;
  int *degree;
  int ns,nr,nq,nx,nss;
  
  // Count number of connections
  degree = drr;
  for (int p = 0;p<m;p++) degree[p] = (degree[p+1] - degree[p]);
  
  // clear the mask of already processed nodes

  X = scinew int[m];
  Q = scinew int[m];
  S = scinew int[m];
  R = scinew int[m];
  
  for (int p=0;p<m;p++) Q[p] = 0;
  
  int root = 0;
  for (int p=0;p<m;p++) if((degree[p] < degree[root])&&(Q[p] == 0)) root = p;

  nss = 0;
  nr = 0;
  ns = 0;
  nq = 0;
  nx = 0;
  
  R[nr++] = root;
  Q[root] = 1; nq++;
  X[nx++] = root;
  
  nr = 0;
  for (int p = rr[root];p<rr[root+1];p++) if (Q[cc[p]] == 0) { R[nr++] = cc[p]; Q[cc[p]] = 1; nq++; }

 

 
  while(1)
  {
    if(nr > 0) 
    { 
      for (int i =0;i<nr;i++)
      {
        int t;
        int k = i;
        // sort entries by order
        for (int j=i+1;j<nr;j++) { if (degree[R[k]] > degree[R[j]]) { k = j;}
        if (k > i)  {t = R[i]; R[i] = R[k]; R[k] = t;}  }
      }
      for (int j=0;j<nr;j++) { X[nx++] = R[j]; S[ns++] = R[j]; }
      nr  = 0;
    }
    if (nx == m) break;
    
    if (nss < ns)
    {
      root = S[nss++];
      nr = 0;
      for (int p = rr[root];p<rr[root+1];p++) if (Q[cc[p]] == 0) { R[nr++] = cc[p]; Q[cc[p]] = 1; nq++; }
    }
    else
    {
      for (int p=0;p<m;p++) if(Q[p] == 0) { root = p; break;}
      for (int p=0;p<m;p++) if((degree[p] < degree[root])&&(Q[p] == 0)) root = p;
      Q[root] = 1; nq++;
      X[nx++] = root;
      nr = 0;
      for (int p = rr[root];p<rr[root+1];p++) if (Q[cc[p]] == 0) { R[nr++] = cc[p]; Q[cc[p]] = 1; nq++; }
    }    
  }

  int t;
  
  // finish mapping matrix and inverse mapping matrix
  if (calcmapping)
  {
    for (int p=0;p<(m+1);p++) { mrr[p] = p; }
    for (int p=0;p<m;p++) { md[p] = 1.0;  }
    for (int p=0;p<m;p++) { mcc[p] = X[p]; }
  }
  
  if (calcimapping)
  {
    for (int p=0;p<(m+1);p++) { imrr[p] = p;}
    for (int p=0;p<m;p++) { imd[p] = 1.0; }
    for (int p=0;p<m;p++) { imcc[X[p]] = p; }
  }
  
   for (int p=0;p<m;p++) { Q[X[p]] = p; }
  
  // finish the reordering new matrix
  int i,j;
  drr[0] = 0;
  i = 0;
  for (int p=0;p<m;p++)
  {
    j = Q[p];
    drr[p+1] = drr[p]+(rr[j+1]-rr[j]);
    for (int r=rr[j];r<rr[j+1];r++)
    {
      dcc[i] = X[cc[r]];
      dd[i] = d[r];
      i++;
    }
  }

  delete[] X;
  delete[] Q;
  delete[] S;
  delete[] R;


}
 
void ExtraMath::mapmatrix(MatrixHandle im,MatrixHandle& om,MatrixHandle mapping)
{ 
  om = mapping*im;
} 
 
}