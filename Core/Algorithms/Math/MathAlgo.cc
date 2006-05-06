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

#include <Core/Algorithms/Math/MathAlgo.h>
#include <Core/Algorithms/Math/BuildFEMatrix.h>

#include <Core/Algorithms/Fields/FieldCount.h>
#include <Core/Datatypes/MatrixOperations.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

MathAlgo::MathAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool MathAlgo::BuildFEMatrix(FieldHandle field, MatrixHandle& matrix, int num_proc, MatrixHandle ConductivityTable, MatrixHandle GeomToComp, MatrixHandle CompToGeom)
{
  BuildFEMatrixAlgo algo;
  
  if(!(algo.BuildFEMatrix(pr_,field,matrix,ConductivityTable,num_proc))) return(false);
  
  if ((GeomToComp.get_rep()==0)&&(CompToGeom.get_rep()==0))
  {
    if (field->is_property("GeomToComp")&&field->is_property("CompToGeom"))
    {
      field->get_property("GeomToComp",GeomToComp);
      field->get_property("CompToGeom",CompToGeom);
      matrix = CompToGeom*matrix*GeomToComp;
    }  
    return (true);
  }
  
  if (CompToGeom.get_rep() == 0) CompToGeom = GeomToComp->transpose();
  if (GeomToComp.get_rep() == 0) GeomToComp = CompToGeom->transpose();
  matrix = GeomToComp*matrix*CompToGeom;
  return (true);
}


bool MathAlgo::ResizeMatrix(MatrixHandle input, MatrixHandle& output, int m, int n)
{ 
  if (input->is_sparse())
  {
    double* val = input->get_val();
    int* row = input->get_row();
    int* col = input->get_col();
    int sm = input->nrows();
    int sn = input->ncols();
    int nnz = input->get_data_size();
 
  
	int newnnz=0;
    for (int p=1; p<(m+1); p++)
    {
      if (p <= sm) for (int q = row[p-1]; q < row[p]; q++) if (col[q] < n) newnnz++;
    }
 
    double* newval = scinew double[newnnz];  
    int* newcol = scinew int[newnnz];
    int* newrow = scinew int[m+1];
    
    if ((newval == 0)||(newcol == 0)||(newrow == 0))
    {
      error("ResizeMatrix: Could not allocate output matrix");
      if (newval) delete newval;
      if (newcol) delete newcol;
      if (newrow) delete newrow;
      return (false);
    }
    
    int k = 0;
    for (int p=0; p<nnz; p++) 
    if (col[p] < n) 
    {
      newval[k] = val[p];
      newcol[k] = col[p];
      k++;
    }
    
    int r=0;
    newrow[0] = 0;
    for (int p=1; p<(m+1); p++)
    {
      if (p <= sm) for (int q = row[p-1]; q < row[p]; q++) if (col[q] < n) r++;
      newrow[p] = r;
    }
    
    output = dynamic_cast<Matrix *>(scinew SparseRowMatrix(m,n,newrow,newcol,newnnz,newval));
    if (output.get_rep() == 0)
    {
      error("ResizeMatrix: Could not allocate output matrix");
      if (newval) delete newval;
      if (newcol) delete newcol;
      if (newrow) delete newrow;
      return (false);    
    }
    return (true);
  }
  else
  {
    MatrixHandle mat = dynamic_cast<Matrix *>(input->dense());
    output = dynamic_cast<Matrix *>(scinew DenseMatrix(m,n));
    if (output.get_rep() == 0)
    {
      error("ResizeMatrix: Could not allocate output matrix");
      return (false);
    }
  
    int sm = input->nrows();
    int sn = input->ncols();
    
    double *src = input->get_data_pointer();
    double *dst = output->get_data_pointer();
    
    int p,q;
    for (p=0;(p<m)&&(p<sm);p++)
    {
      for (q=0;(q<n)&&(q<sn);q++)
      {
        dst[q+p*n] = src[q+p*sn];
      }
      for (;q<n;q++) dst[q+p*n] = 0.0;
    }
    for (;p<m;p++)
      for (q=0;q<n;q++) dst[q+p*n]= 0.0;
   
    return (true);
  }
  
  return (false);
}

bool MathAlgo::CreateSparseMatrix(SparseElementVector& input, MatrixHandle& output, int m, int n)
{
  std::sort(input.begin(),input.end());
  
  int nnz = 1;
  int q = 0;
  for (int p=1; p < input.size(); p++)
  {
    if (input[p] == input[q])
    {
      input[q].val += input[p].val; 
	  input[p].val = 0.0;
    }
    else
    {
      nnz++;
      q=p;
    }
  }
  
  // reserve memory
  
  int *rows = scinew int[m+1];
  int *cols = scinew int[nnz];
  double *vals = scinew double[nnz];
  
  if ((rows == 0)||(cols == 0)||(vals == 0))
  {
    if (rows) delete[] rows;
    if (cols) delete[] cols;
    if (vals) delete[] vals;
    error("CreateSparseMatrix: Could not allocate memory for matrix");
    return (false);
  }
  
  rows[0] = 0;
  q = 0;
  
  int k = 0;
  for (int p=0; p < m; p++)
  {
    while ((q < input.size())&&(input[q].row == p)) 
	{ 
	  if (input[q].val)
	  {
	    cols[k] = input[q].col; 
	    vals[k] = input[q].val;
		k++; 
	  }
	  q++; 
	}
    rows[p+1] = k;
  }   
  
  output = dynamic_cast<Matrix *>(scinew SparseRowMatrix(m,n,rows,cols,nnz,vals));
  if (output.get_rep()) return (true);
  
  return (false);
}



bool MathAlgo::ReverseCuthillmcKee(MatrixHandle im,MatrixHandle& om,MatrixHandle& mapping,bool calcmapping)
{
  int *rr, *cc;
  double *d;
  int n,m,nnz;
  
  if (im.get_rep() == 0)
  {
    error("ReverseCuthillmcKee: No input matrix was found");
    return (false);
  }
  
  if (im->ncols() != im->nrows())
  {
    error("ReverseCuthillmcKee: Matrix is not square");
    return (false);  
  }
  
  if (im->is_sparse() == false) 
  {
    error("ReverseCuthillmcKee: Matrix is not sparse");
    return (false);
  }
  SparseRowMatrix* sim = im->as_sparse();
  
  om = sim->clone();
  if (om.get_rep() == 0)
  {
    error("ReverseCuthillmcKee: Could not copy sparse matrix");
    return (false);  
  }
  
  m  = sim->nrows();
  n  = sim->ncols();
  nnz = sim->nnz;
  rr = sim->rows;
  cc = sim->columns;
  d  = sim->a;
  
  // reserve mapping space  
  int *mrr, *mcc;
  double *md;

  int *drr, *dcc;
  double *dd;

  if (calcmapping)
  {
    mrr = scinew int[m+1];
    mcc = scinew int[m];
    md  = scinew double[m];
    if ((mrr == 0)||(mcc == 0)||(md == 0))
    {
      if (mrr ) delete[] mrr;
      if (mcc ) delete[] mcc;
      if (md ) delete[] md;      

      error("ReverseCuthillmcKee: Could not reserve space for mapping matrix");    
      return (false);
    }
    
    mapping = scinew SparseRowMatrix(m,m,mrr,mcc,m,md);
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
  
  if ((X==0)||(Q==0)||(S==0)||(R==0))
  {
    if (X) delete[] X;
    if (Q) delete[] Q;
    if (S) delete[] S;
    if (R) delete[] R;
    error("ReverseCuthillmcKee: Could not allocate enough memory");
    return (false);
  }
  
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

  return (true);
} 
 

bool MathAlgo::CuthillmcKee(MatrixHandle im,MatrixHandle& om,MatrixHandle& mapping,bool calcmapping)
{
 int *rr, *cc;
  double *d;
  int n,m,nnz;

  if (im.get_rep() == 0)
  {
    error("ReverseCuthillmcKee: No input matrix was found");
    return (false);
  }
  
  if (im->ncols() != im->nrows())
  {
    error("ReverseCuthillmcKee: Matrix is not square");
    return (false);  
  }
  
  if (im->is_sparse() == false) 
  {
    error("CuthillmcKee: Matrix is not sparse");
    return (false);
  }
  
  SparseRowMatrix* sim = im->as_sparse();
  
  om = sim->clone();
  if (om.get_rep() == 0)
  {
    error("CuthillmcKee: Could not copy sparse matrix");
    return (false);  
  }
  
  m  = sim->nrows();
  n  = sim->ncols();
  nnz = sim->nnz;
  rr = sim->rows;
  cc = sim->columns;
  d  = sim->a;
  
  // reserve mapping space
  
  int *mrr, *mcc;
  double *md;

  int *drr, *dcc;
  double *dd;

  if (calcmapping)
  {
    mrr = scinew int[m+1];
    mcc = scinew int[m];
    md  = scinew double[m];
    
    if ((mrr == 0)||(mcc == 0)||(md == 0))
    {
      if (mrr ) delete[] mrr;
      if (mcc ) delete[] mcc;
      if (md ) delete[] md;      

      error("CuthillmcKee: Could not reserve space for mapping matrix");    
      return (false);
    }    
    
    mapping = scinew SparseRowMatrix(m,m,mrr,mcc,m,md);
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

  if ((X==0)||(Q==0)||(S==0)||(R==0))
  {
    if (X) delete[] X;
    if (Q) delete[] Q;
    if (S) delete[] S;
    if (R) delete[] R;
    error("CuthillmcKee: Could not allocate enough memory");
    return (false);
  }
    
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


  return (true);
}

bool MathAlgo::ApplyRowOperation(MatrixHandle input, MatrixHandle& output, std::string method)
{
  if (input.get_rep() == 0)
  {
    error("ApplyRowOperation: no input matrix found");
    return (false);
  }
  
  int nrows = input->nrows();
  int ncols = input->ncols();
  
  output = dynamic_cast<Matrix*>(scinew DenseMatrix(nrows,1));
  if (output.get_rep() == 0)
  {
    error("ApplyRowOperation: could not create output matrix");
    return (false);  
  }

  double *dest = output->get_data_pointer();
  for (int q=0; q<nrows; q++) dest[q] = 0.0;

  if (input->is_sparse())
  {
    int *rows = input->sparse()->rows;
    int *cols = input->sparse()->columns;
    double *vals = input->sparse()->a;
    
    if (method == "Sum")
    {
      for (int q=0; q<nrows; q++) { for (int r = rows[q]; r < rows[q+1]; r++) dest[q] += vals[r]; }        
    }
    else if ((method == "Average")||(method == "Mean"))
    {
      for (int q=0; q<nrows; q++) 
      {
        for (int r = rows[q]; r < rows[q+1]; r++) dest[q] += vals[r];  
        dest[q] /= static_cast<double>(ncols);  
      }            
    }
    else if (method == "Norm")
    {
      for (int q=0; q<nrows; q++) 
      {
        for (int r = rows[q]; r < rows[q+1]; r++) dest[q] += (vals[r])*(vals[r]);  
        if (rows[q] != rows[q+1]) dest[q] = sqrt(dest[q]);  
      }                
    }
    else if (method == "Variance")
    {
      double mean;
      for (int q=0; q<nrows; q++) 
      {
        mean = 0.0;
        for (int r = rows[q]; r < rows[q+1]; r++) mean += vals[r];
        if (ncols) mean /= static_cast<double>(ncols);
        for (int r = rows[q]; r < rows[q+1]; r++) dest[q] += (vals[r]-mean)*(vals[r]-mean);
        dest[q] += (ncols-(rows[q+1]-rows[q]))*mean*mean;  
        if (ncols > 1) dest[q] = dest[q]/static_cast<double>(ncols-1); else dest[q] = 0.0;
      }               
    }
    else if (method == "StdDev")
    {
      double mean;
      for (int q=0; q<nrows; q++) 
      {
        mean = 0.0;
        for (int r = rows[q]; r < rows[q+1]; r++) mean += vals[r];
        if (ncols) mean /= static_cast<double>(ncols);
        for (int r = rows[q]; r < rows[q+1]; r++) dest[q] += (vals[r]-mean)*(vals[r]-mean);  
        dest[q] += (ncols-(rows[q+1]-rows[q]))*mean*mean;  
        if (ncols > 1) dest[q] = sqrt(dest[q]/static_cast<double>(ncols-1)); else dest[q] = 0.0;
      }               
    }
    else if (method == "Maximum")
    {
      for (int q=0; q<nrows; q++) 
      {
        if (rows[q+1]-rows[q] == ncols) dest[q] = -DBL_MAX; else dest[q] = 0.0;
        for (int r = rows[q]; r < rows[q+1]; r++) if (vals[r] > dest[q]) dest[q] = vals[r];
      }                   
    }
    else if (method == "Minimum")
    {
      for (int q=0; q<nrows; q++) 
      {
        if (rows[q+1]-rows[q] == ncols) dest[q] = DBL_MAX; else dest[q] = 0.0;
        for (int r = rows[q]; r < rows[q+1]; r++) if (vals[r] < dest[q]) dest[q] = vals[r];
      }                   
    }
    else if (method == "Median")
    {
      for (int q=0; q<nrows; q++) 
      {
        std::vector<double> dpos;
        std::vector<double> dneg;
        for (int r = rows[q]; r < rows[q+1]; r++) if (vals[r] < 0.0 ) dneg.push_back(vals[r]); else dpos.push_back(vals[r]);
        if (dpos.size() >= ncols/2) std::sort(dpos.begin(),dpos.end());
        if (dneg.size() >= ncols/2) std::sort(dneg.begin(),dneg.end());
        if (2*(ncols/2) == ncols)
        {
          double val1 = 0.0;
          double val2 = 0.0;
          int q1 = (ncols/2)-1;
          int q2 = ncols/2; 
          if ( q1 < dneg.size()) val1 = dneg[q1];
          if ( q2 < dneg.size()) val2 = dneg[q2];
          if ( (ncols-1)-q1 < dpos.size()) val1 = dpos[dpos.size()-ncols+q1];
          if ( (ncols-1)-q2 < dpos.size()) val2 = dpos[dpos.size()-ncols+q2];          
          dest[q] = 0.5*(val1+val2);
        }
        else
        {
          double val1 = 0.0;
          int q1 = (ncols/2);
          if ( q1 < dneg.size()) val1 = dneg[q1];
          if ( (ncols-1)-q1 < dpos.size()) val1 = dpos[dpos.size()-ncols+q1];
          dest[q] = val1;
        }       
      }                   
    }
    else
    {
      error ("ApplyRowOperation: This method has not yet been implemented");
      return (false);
    }
  }
  else
  {
    DenseMatrix* mat = input->dense();
    
    int m = mat->nrows();
    int n = mat->ncols();
    double* data = mat->get_data_pointer();
  
    if (method == "Sum")
    {
      int k = 0;
      for (int p=0; p<m; p++) dest[p] = 0.0;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++) dest[p] += data[k++]; 
    }
    else if (method == "Mean")
    {
      int k = 0;
      for (int p=0; p<m; p++) dest[p] = 0.0;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++) dest[p] += data[k++]; 
      for (int p=0; p<m; p++) dest[p] /= static_cast<double>(n);  
    }
    else if (method == "Variance")
    {
      std::vector<double> mean(m);
      int k = 0;
      for (int p=0; p<m; p++) dest[p] = 0.0;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++) mean[p] += data[k++];
      for (int p=0; p<m; p++) mean[p] /= static_cast<double>(n);  
      k = 0;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++, k++) dest[p] += (data[k]-mean[p])*(data[k]-mean[p]); 
      for (int p=0; p<m; p++)  if (n > 1) dest[p] /= static_cast<double>(n-1);  else dest[p] = 0.0;
    }
    else if (method == "StdDev")
    {
      std::vector<double> mean(m);
      int k = 0;
      for (int p=0; p<m; p++) dest[p] = 0.0;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++) mean[p] += data[k++];
      for (int p=0; p<m; p++) mean[p] /= static_cast<double>(n);  
      k = 0;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++, k++) dest[p] += (data[k]-mean[p])*(data[k]-mean[p]); 
      for (int p=0; p<m; p++)  if (n > 1) dest[p] = sqrt(dest[p]/static_cast<double>(n-1));  else dest[p] = 0.0;
    }
    else if (method == "Norm")
    {
      int k = 0;
      for (int p=0; p<m; p++) dest[p] = 0.0;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++, k++) dest[p] += data[k]*data[k]; 
      for (int p=0; p<m; p++) dest[p] = sqrt(dest[p]); 
    }
    else if (method == "Maximum")
    {
      int k = 0;
      for (int p=0; p<m; p++) dest[p] = -DBL_MAX;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++, k++) if (data[k] > dest[p]) dest[p] = data[k]; 
    }
    else if (method == "Minimum")
    {
      int k = 0;
      for (int p=0; p<m; p++) dest[p] = DBL_MAX;
      for (int p=0; p<m; p++) for (int q=0; q<n; q++, k++) if (data[k] < dest[p]) dest[p] = data[k]; 
    }
    else if (method == "Median")
    {
      int k = 0;
      std::vector<double> v(n);

      for (int p=0; p<m; p++)
      {
        for (int q=0; q<n; q++, k++) v[q] = data[k];
        std::sort(v.begin(),v.end());
        if ((n/2)*2 == n)
        {
          dest[p] = 0.5*(v[n/2]+v[(n/2) -1]);
        }
        else
        {
          dest[p] = v[n/2];
        }
      }
    }
    else
    {
      error("ApplyRowOperation: This method has not yet been implemented");
      return (false);    
    }
  }
  
  return (true);
}

bool MathAlgo::ApplyColumnOperation(MatrixHandle input, MatrixHandle& output, std::string method)
{
  if (input.get_rep() == 0)
  {
    error("ApplyRowOperation: no input matrix found");
    return (false);
  }
  MatrixHandle t = dynamic_cast<Matrix *>(input->transpose());
  if(!(ApplyRowOperation(t,t,method))) return (false);
  output = dynamic_cast<Matrix *>(t->transpose());
  return (true);
} 


} // end SCIRun namespace
