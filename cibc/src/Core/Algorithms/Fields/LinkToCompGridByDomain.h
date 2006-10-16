/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


#ifndef CORE_ALGORITHMS_FIELDS_LINKTOCOMPGRIDBYDOMAIN_H
#define CORE_ALGORITHMS_FIELDS_LINKTOCOMPGRIDBYDOMAIN_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;


class LinkToCompGridByDomainAlgo : public DynamicAlgoBase
{
public:
  virtual bool LinkToCompGridByDomain(ProgressReporter *pr, FieldHandle input, MatrixHandle NodeLink, MatrixHandle& GeomToComp, MatrixHandle& CompToGeom);

};

template <class FSRC>
class LinkToCompGridByDomainAlgoT : public LinkToCompGridByDomainAlgo
{
public:
  virtual bool LinkToCompGridByDomain(ProgressReporter *pr, FieldHandle input, MatrixHandle NodeLink, MatrixHandle& GeomToComp, MatrixHandle& CompToGeom);
};



template <class FSRC>
bool LinkToCompGridByDomainAlgoT<FSRC>::LinkToCompGridByDomain(ProgressReporter *pr, FieldHandle input, MatrixHandle NodeLink, MatrixHandle& GeomToComp, MatrixHandle& CompToGeom)
{
  if (NodeLink.get_rep() == 0)
  {
    pr->error("LinkToCompGrid: No matrix on input");
    return (false);
  }

  if (!(NodeLink->is_sparse()))
  {
    pr->error("LinkToCompGrid: NodeLink Matrix is not sparse");
    return (false);  
  }

  if (NodeLink->nrows() != NodeLink->ncols())
  {
    pr->error("LinkToCompGrid: NodeLink Matrix needs to be square");
    return (false);      
  }

  FSRC *ifield = dynamic_cast<FSRC*>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("LinkToCompGrid: FieldHandle is empty");
    return (false);     
  }
  
  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh.get_rep() == 0)
  {
    pr->error("LinkToCompGrid: Field does not have a mesh");
    return (false);       
  }
        
  typename FSRC::mesh_type::Node::size_type numnodes;
  imesh->size(numnodes);
  
  if (NodeLink->nrows() != numnodes)
  {
    pr->error("LinkToCompGrid: NodeLink Matrix has improper dimensions");
    return (false);        
  }
  
  SparseRowMatrix* spr = dynamic_cast<SparseRowMatrix*>(NodeLink.get_rep());
  int m = spr->ncols();
  int *rows = spr->rows;
  int *cols = spr->columns;
  double *vals = spr->a;
  
  int *rr = scinew int[m+1];
  int *cc = scinew int[m];
  double *vv = scinew double[m];  
  if ((rr == 0)||(cc == 0)||(vv == 0))
  {
    if (rr) delete[] rr;
    if (cc) delete[] cc;
    if (vv) delete[] vv;
    
    pr->error("LinkToCompGrid: Could not allocate memory for sparse matrix");
    return (false);        
  }
  
  imesh->synchronize(Mesh::NODE_NEIGHBORS_E);
  
  for (int r=0; r<m; r++) rr[r] = r;

  for (int r=0; r<m; r++)
  {
    for (int c=rows[r]; c<rows[r+1]; c++)
    {
      if (cols[c] > r) 
      {
        typename FSRC::mesh_type::Elem::array_type elems1;
        typename FSRC::mesh_type::Elem::array_type elems2;
        typename FSRC::mesh_type::Node::index_type idx1;
        typename FSRC::mesh_type::Node::index_type idx2;
        imesh->to_index(idx1,r);
        imesh->get_elems(elems1,idx1);
        imesh->to_index(idx2,cols[c]);
        imesh->get_elems(elems2,idx2);
        if (ifield->value(elems1[0]) == ifield->value(elems2[0])) rr[cols[c]] = r;
      }
    }
  }

  for (int r=0; r< m; r++)
  {
    int p = r;
    while (rr[p] != p) p = rr[p];
    rr[r] = p;      
  }

  int k=0;
  for (int r=0; r<m; r++)
  {
    if (rr[r] == r) 
    {
      rr[r] = k++;
    }
    else
    {
      rr[r] = rr[rr[r]];
    }
  }

  for (int r = 0; r < m; r++)
  {
    cc[r] = rr[r];
    rr[r] = r;
    vv[r] = 1.0;
  }
  rr[m] = m; // An extra entry goes on the end of rr.

  spr = scinew SparseRowMatrix(m, k, rr, cc, m, vv);

  if (spr == 0)
  {
    pr->error("LinkToCompGrid: Could build geometry to computational mesh mapping matrix");
    return (false);
  }

  CompToGeom = spr;
  GeomToComp = spr->transpose();

  if ((GeomToComp.get_rep() == 0)||(CompToGeom.get_rep() == 0))
  {
    pr->error("LinkToCompGrid: Could build geometry to computational mesh mapping matrix");
    return (false);
  }
  
  return (true);
}

} // end namespace SCIRunAlgo

#endif 
