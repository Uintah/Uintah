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


//    File   : ChangeFieldDataAt.h
//    Author : McKay Davis
//    Date   : July 2002


#if !defined(ChangeFieldDataAt_h)
#define ChangeFieldDataAt_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/SparseRowMatrix.h>


namespace SCIRun {


class ChangeFieldDataAtAlgoCreate : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      MatrixHandle &interp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FSRC>
class ChangeFieldDataAtAlgoCreateT : public ChangeFieldDataAtAlgoCreate
{
public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      MatrixHandle &interp);
};


template <class FSRC>
FieldHandle
ChangeFieldDataAtAlgoCreateT<FSRC>::execute(ProgressReporter *mod,
					    FieldHandle fsrc_h,
					    Field::data_location at,
					    MatrixHandle &interp)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());
  typename FSRC::mesh_handle_type mesh = fsrc->get_typed_mesh();

  // Create the field with the new mesh and data location.
  FSRC *fout = scinew FSRC(fsrc->get_typed_mesh(), at);
  fout->resize_fdata();

  if (fsrc->data_at() == Field::NODE)
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

    if (at == Field::EDGE)
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
    else if (at == Field::FACE)
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
    else if (at == Field::CELL)
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
      interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);
    }
    else if (rr)
    {
      delete rr;
    }
  }
  else if (at == Field::NODE && fsrc->data_at() == Field::CELL)
  {
    typename FSRC::mesh_type::Cell::size_type nsize;
    mesh->size(nsize);
    const int ncols = nsize;

    typename FSRC::mesh_type::Node::size_type osize;
    mesh->size(osize);
    const int nrows = osize;

    int *rr = scinew int[nrows + 1];
    vector<int> cctmp;
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
  else if (at == Field::NODE && fsrc->data_at() == Field::FACE)
  {
    typename FSRC::mesh_type::Face::size_type nsize;
    mesh->size(nsize);
    const int ncols = nsize;

    typename FSRC::mesh_type::Node::size_type osize;
    mesh->size(osize);
    const int nrows = osize;

    int *rr = scinew int[nrows + 1];
    vector<int> cctmp;
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
  else if (at == Field::NODE && fsrc->data_at() == Field::EDGE)
  {
    typename FSRC::mesh_type::Edge::size_type nsize;
    mesh->size(nsize);
    const int ncols = nsize;

    typename FSRC::mesh_type::Node::size_type osize;
    mesh->size(osize);
    const int nrows = osize;

    int *rr = scinew int[nrows + 1];
    vector<int> cctmp;
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

  fout->copy_properties(fsrc);

  return fout;
}


} // end namespace SCIRun

#endif // ChangeFieldDataAt_h
