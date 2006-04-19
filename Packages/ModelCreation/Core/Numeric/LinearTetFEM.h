//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.

#ifndef MODELCREATION_CORE_NUMERIC_LINEARTETFEM_H
#define MODELCREATION_CORE_NUMERIC_LINEARTETFEM_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class SparseElem {
public:
  int     row;
  int     col;
  double  val;
};

inline bool operator==(const SparseElem& s1,const SparseElem& s2)
{
  if ((s1.row == s2.row)&&(s1.col == s2.col)) return (true);
  return (false);
}    

inline bool operator<(const SparseElem& s1, const SparseElem& s2)
{
  if (s1.row < s2.row) return(true);
  if (s1.row == s2.row) if (s1.col < s2.col) return(true);
  return (false);
}


class LinearTetFEMAlgo : public DynamicAlgoBase
{
public:
  virtual bool LinearTetFEM(ProgressReporter *pr, FieldHandle input, MatrixHandle& spr, MatrixHandle conductivity);
};

template<class FSRC>
class LinearTetFEMAlgoT : public LinearTetFEMAlgo
{
public:
  virtual bool LinearTetFEM(ProgressReporter *pr, FieldHandle input, MatrixHandle& spr, MatrixHandle conductivity);
};

template<class FSRC>
bool LinearTetFEMAlgoT<FSRC>::LinearTetFEM(ProgressReporter *pr, FieldHandle input, MatrixHandle& output, MatrixHandle conductivity)
{
  FSRC* ifield = dynamic_cast<FSRC* >(input.get_rep());
  if (ifield == 0)
  {
    pr->error("DistanceField: There is no input field");
    return(false);
  }

  std::vector<std::pair<string, Tensor> > tensors;
  ifield->get_property("conductivity_table",tensors);

  if ((tensors.size() == 0)&&(conductivity.get_rep()))
  {
    DenseMatrix* mat = conductivity->dense();
    if (mat)
    {
      double* data = mat->get_data_pointer();
      int m = mat->nrows();
      Tensor T; 

      if (mat->ncols() == 1)
      {
        for (int p=0; p<m;p++)
        {
          T.mat_[0][0] = data[0*m+p];
          T.mat_[1][0] = 0.0;
          T.mat_[2][0] = 0.0;
          T.mat_[0][1] = 0.0;
          T.mat_[1][1] = data[0*m+p];
          T.mat_[2][1] = 0.0;
          T.mat_[0][2] = 0.0;
          T.mat_[1][2] = 0.0;
          T.mat_[2][2] = data[0*m+p];
          tensors.push_back(std::pair<string, Tensor>("",T));
        }
      }
       
      if (mat->ncols() == 6)
      {
        for (int p=0; p<m;p++)
        {
          T.mat_[0][0] = data[0*m+p];
          T.mat_[1][0] = data[1*m+p];
          T.mat_[2][0] = data[2*m+p];
          T.mat_[0][1] = data[1*m+p];
          T.mat_[1][1] = data[3*m+p];
          T.mat_[2][1] = data[4*m+p];
          T.mat_[0][2] = data[2*m+p];
          T.mat_[1][2] = data[4*m+p];
          T.mat_[2][2] = data[5*m+p];
          tensors.push_back(std::pair<string, Tensor>("",T));
        }
      }

      if (mat->ncols() == 9)
      {
        for (int p=0; p<m;p++)
        {
          T.mat_[0][0] = data[0*m+p];
          T.mat_[1][0] = data[1*m+p];
          T.mat_[2][0] = data[2*m+p];
          T.mat_[0][1] = data[1*m+p];
          T.mat_[1][1] = data[4*m+p];
          T.mat_[2][1] = data[5*m+p];
          T.mat_[0][2] = data[2*m+p];
          T.mat_[1][2] = data[5*m+p];
          T.mat_[2][2] = data[8*m+p];
          tensors.push_back(std::pair<string, Tensor>("",T));
        }
      }
    }
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FSRC::mesh_type::Elem::iterator it, it_end;
  typename FSRC::mesh_type::Node::array_type nodes;
  typename FSRC::mesh_type::Elem::size_type numelems; 


  imesh->size(numelems);
  std::vector<SparseElem> sprelems(numelems*16);
  imesh->begin(it);
  imesh->end(it_end);
  
  int k = 0;
  while (it != it_end)
  {
    typename FSRC::value_type T;
    Point y1,y2,y3,y4;
    Vector a,b,n;
    double alpha;
    double vol;
    Vector gN[4];
    
    imesh->get_nodes(nodes,*it);
    imesh->get_center(y1,nodes[0]);
    imesh->get_center(y2,nodes[1]);
    imesh->get_center(y3,nodes[2]);
    imesh->get_center(y4,nodes[3]);
    T = ifield->value(*it);
  
    a = y2-y3;
    b = y2-y4;
    n = Cross(a,b);
    alpha = 1/Dot(n,Vector(y2-y1));
    gN[0] = alpha*n;
    
    a = y3-y4;
    b = y3-y1;
    n = Cross(a,b);
    alpha = 1/Dot(n,Vector(y3-y2));
    gN[1] = alpha*n;
    
    a = y4-y1;
    b = y4-y2;
    n = Cross(a,b);
    alpha = 1/Dot(n,Vector(y4-y3));
    gN[2] = alpha*n;

    a = y1-y2;
    b = y1-y3;
    n = Cross(a,b);
    alpha = 1/Dot(n,Vector(y1-y4));
    gN[3] = alpha*n;
  
    vol = fabs((1.0/6.0)*Dot(n,Vector(y1-y4)));
  
    SparseElem item;
    
    if (tensors.size() == 0)
    {
      for (int i=0;i<4;i++)
      {
        for (int j=0;j<4;j++)
        {
          item.row = nodes[i]; 
          item.col = nodes[j];
          item.val = static_cast<double>(vol*Dot(gN[i],T*gN[j]));
          sprelems[k++] = item;
        }
      }
    }
    else
    {
      for (int i=0;i<4;i++)
      {
        for (int j=0;j<4;j++)
        {
          item.row = nodes[i]; 
          item.col = nodes[j];
          item.val = static_cast<double>(vol*Dot(gN[i],tensors[static_cast<int>(T)].second*gN[j]));
          sprelems[k++] = item;
        }
      }    
    }
    
    ++it;
  }
  
  typename FSRC::mesh_type::Node::size_type numnodes;
  imesh->size(numnodes);
  int m = numnodes;
  
  std::sort(sprelems.begin(),sprelems.end());
  
  int nnz = 1;
  int q = 0;
  for (int p=1; p < sprelems.size(); p++)
  {
    if (sprelems[p] == sprelems[q])
    {
      sprelems[q].val += sprelems[p].val; 
      sprelems[p].val = 0;
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
    pr->error("CreateSparseMatrix: Could not allocate memory for matrix");
    return (false);
  }
  
  rows[0] = 0;
  q = 0;
  
  k = 0;
  for (int p=0; p < m; p++)
  {
    while ((k < sprelems.size())&&(sprelems[k].row <= p)) {  if (sprelems[k].val) { cols[q] = sprelems[k].col; vals[q] = sprelems[k].val; q++;} k++; }
    rows[p+1] = q;
  }   
  
  
  output = dynamic_cast<Matrix *>(scinew SparseRowMatrix(m,m,rows,cols,nnz,vals));
  if (output.get_rep()) return (true);
  
  return (false);
}

}

#endif


