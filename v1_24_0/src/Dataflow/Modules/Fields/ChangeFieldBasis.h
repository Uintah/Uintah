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



//    File   : ChangeFieldBasis.h
//    Author : McKay Davis
//    Date   : July 2002


#if !defined(ChangeFieldBasis_h)
#define ChangeFieldBasis_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <algorithm>

namespace SCIRun {


class ChangeFieldBasisAlgoCreate : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle fsrc_h,
			      int basis_order,
			      MatrixHandle &interp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FSRC>
class ChangeFieldBasisAlgoCreateT : public ChangeFieldBasisAlgoCreate
{
public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle fsrc_h,
			      int basis_order,
			      MatrixHandle &interp);
};


template <class FSRC>
FieldHandle
ChangeFieldBasisAlgoCreateT<FSRC>::execute(ProgressReporter *mod,
					    FieldHandle fsrc_h,
					    int basis_order,
					    MatrixHandle &interp)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());
  typename FSRC::mesh_handle_type mesh = fsrc->get_typed_mesh();

  // Create the field with the new mesh and data location.
  FSRC *fout = scinew FSRC(fsrc->get_typed_mesh(), basis_order);
  fout->resize_fdata();

  if (fsrc->basis_order() > 0)
  {
    typename FSRC::mesh_type::Node::size_type nodesize;
    mesh->size(nodesize);
    const int ncols = nodesize;

    int nrows;
    int *rr = 0;
    int nnz = 0;
    int *cc = 0;
    double *d = 0;
    typename FSRC::mesh_type::Node::array_type tmparray;

    if (basis_order == 0 && mesh->dimensionality() == 1)
    {
      typename FSRC::mesh_type::Edge::size_type osize;
      mesh->size(osize);
      nrows = osize;

      rr = scinew int[nrows+1];
      rr[0] = 0;
      size_t counter = 0;
      typename FSRC::mesh_type::Edge::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      while (itr != eitr)
      {
	mesh->get_nodes(tmparray, *itr);
	const int mult = tmparray.size();
	if (counter == 0)
	{
	  nnz = nrows*mult;
	  cc = scinew int[nnz];
	  d = scinew double[nnz];
	}
	for (int i = 0; i < mult; i++)
	{
	  cc[counter*mult + i] = tmparray[i];
	  d[counter*mult + i] = 1.0 / mult;
	}

	++itr;
	++counter;
	rr[counter] = rr[counter-1] + mult;
      }
    } 
    else if (basis_order == 0 && mesh->dimensionality() == 2)
    {
     typename FSRC::mesh_type::Face::size_type osize;
      mesh->size(osize);
      nrows = osize;

      rr = scinew int[nrows+1];
      rr[0] = 0;
      size_t counter = 0;
      typename FSRC::mesh_type::Face::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      while (itr != eitr)
      {
	mesh->get_nodes(tmparray, *itr);
	const int mult = tmparray.size();
	if (counter == 0)
	{
	  nnz = nrows*mult;
	  cc = scinew int[nnz];
	  d = scinew double[nnz];
	}
	for (int i = 0; i < mult; i++)
	{
	  cc[counter*mult + i] = tmparray[i];
	  d[counter*mult + i] = 1.0 / mult;
	}

	++itr;
	++counter;
	rr[counter] = rr[counter-1] + mult;
      }
    }    
    else if (basis_order == 0 && mesh->dimensionality() == 3)
    {
      typename FSRC::mesh_type::Cell::size_type osize;
      mesh->size(osize);
      nrows = osize;

      rr = scinew int[nrows+1];
      rr[0] = 0;
      size_t counter = 0;
      typename FSRC::mesh_type::Cell::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      while (itr != eitr)
      {
	mesh->get_nodes(tmparray, *itr);
	const int mult = tmparray.size();
	if (counter == 0)
	{
	  nnz = nrows*mult;
	  cc = scinew int[nnz];
	  d = scinew double[nnz];
	}
	for (int i = 0; i < mult; i++)
	{
	  cc[counter*mult + i] = tmparray[i];
	  d[counter*mult + i] = 1.0 / mult;
	}

	++itr;
	++counter;
	rr[counter] = rr[counter-1] + mult;
      }
    }

    if (rr && cc)
    {
      for (int i = 0; i < nrows; i++)
      {
        std::sort(cc + rr[i], cc + rr[i+1]);
      }
      interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);
    }
    else if (rr)
    {
      delete rr;
    }
  }
  try {
    if (basis_order == 1 && fsrc->basis_order() == 0 && 
	mesh->dimensionality() == 3)
    {
      mesh->synchronize(Mesh::NODE_NEIGHBORS_E);

      typename FSRC::mesh_type::Cell::size_type nsize;
      mesh->size(nsize);
      const int ncols = (int)nsize;

      typename FSRC::mesh_type::Node::size_type osize;
      mesh->size(osize);
      const int nrows = (int)osize;

      int *rr = scinew int[nrows + 1];
      vector<unsigned int> cctmp;
      vector<double> dtmp;

      typename FSRC::mesh_type::Cell::array_type tmparray;
      typename FSRC::mesh_type::Node::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      rr[0] = 0;
      int counter = 0;
      while (itr != eitr)
      {
	mesh->get_cells(tmparray, *itr);
	for (unsigned int i = 0; i < tmparray.size(); i++)
	{
	  cctmp.push_back(tmparray[i]);
	  dtmp.push_back(1.0 / tmparray.size()); // Weight by distance?
	}

	++itr;
	++counter;
	rr[counter] = rr[counter-1] + tmparray.size();
      }

      const int nnz = cctmp.size();
      int *cc = scinew int[nnz];
      double *d = scinew double[nnz];
      for (int i = 0; i < nnz; i++)
      {
	cc[i] = cctmp[i];
	d[i] = dtmp[i];
      }

      interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);
    }    
    else if (basis_order == 1 && fsrc->basis_order() == 0 && 
	     mesh->dimensionality() == 2)
    {
      mesh->synchronize(Mesh::NODE_NEIGHBORS_E);

      typename FSRC::mesh_type::Face::size_type nsize;
      mesh->size(nsize);
      const int ncols = (int)nsize;

      typename FSRC::mesh_type::Node::size_type osize;
      mesh->size(osize);
      const int nrows = (int)osize;

      int *rr = scinew int[nrows + 1];
      vector<unsigned int> cctmp;
      vector<double> dtmp;

      typename FSRC::mesh_type::Face::array_type tmparray;
      typename FSRC::mesh_type::Node::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      rr[0] = 0;
      int counter = 0;
      while (itr != eitr)
      {
	mesh->get_faces(tmparray, *itr);
	for (unsigned int i = 0; i < tmparray.size(); i++)
	{
	  cctmp.push_back(tmparray[i]);
	  dtmp.push_back(1.0 / tmparray.size()); // Weight by distance?
	}

	++itr;
	++counter;
	rr[counter] = rr[counter-1] + tmparray.size();
      }

      const int nnz = cctmp.size();
      int *cc = scinew int[nnz];
      double *d = scinew double[nnz];
      for (int i = 0; i < nnz; i++)
      {
	cc[i] = cctmp[i];
	d[i] = dtmp[i];
      }
    
      interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);
    }
    else if (basis_order == 1 && fsrc->basis_order() == 0 && 
	     mesh->dimensionality() == 1)
    {
      mesh->synchronize(Mesh::NODE_NEIGHBORS_E);

      typename FSRC::mesh_type::Edge::size_type nsize;
      mesh->size(nsize);
      const int ncols = (int)nsize;

      typename FSRC::mesh_type::Node::size_type osize;
      mesh->size(osize);
      const int nrows = (int)osize;

      int *rr = scinew int[nrows + 1];
      vector<unsigned int> cctmp;
      vector<double> dtmp;

      typename FSRC::mesh_type::Edge::array_type tmparray;
      typename FSRC::mesh_type::Node::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      rr[0] = 0;
      int counter = 0;
      while (itr != eitr)
      {
	mesh->get_edges(tmparray, *itr);
	for (unsigned int i = 0; i < tmparray.size(); i++)
	{
	  cctmp.push_back(tmparray[i]);
	  dtmp.push_back(1.0 / tmparray.size()); // Weight by distance?
	}

	++itr;
	++counter;
	rr[counter] = rr[counter-1] + tmparray.size();
      }

      const int nnz = cctmp.size();
      int *cc = scinew int[nnz];
      double *d = scinew double[nnz];
      for (int i = 0; i < nnz; i++)
      {
	cc[i] = cctmp[i];
	d[i] = dtmp[i];
      }

      interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);
    }
  } catch (...)
  {
  }

  fout->copy_properties(fsrc);

  return fout;
}


} // end namespace SCIRun

#endif // ChangeFieldBasis_h
