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
    int nrows;
    int mult;
    typename FSRC::mesh_type::Node::array_type tmparray;
    if (at == Field::EDGE)
    {
      typename FSRC::mesh_type::Edge::size_type osize;
      mesh->size(osize);
      nrows = osize;
      mesh->get_nodes(tmparray, typename FSRC::mesh_type::Edge::index_type(0));
      mult = tmparray.size();
    }
    else if (at == Field::FACE)
    {
      typename FSRC::mesh_type::Face::size_type osize;
      mesh->size(osize);
      nrows = osize;
      mesh->get_nodes(tmparray, typename FSRC::mesh_type::Face::index_type(0));
      mult = tmparray.size();
    }
    else if (at == Field::CELL)
    {
      typename FSRC::mesh_type::Cell::size_type osize;
      mesh->size(osize);
      nrows = osize;
      mesh->get_nodes(tmparray, typename FSRC::mesh_type::Cell::index_type(0));
      mult = tmparray.size();
    }

    typename FSRC::mesh_type::Node::size_type nodesize;
    mesh->size(nodesize);
    const int ncols = nodesize;

    cout << "rows=" << nrows << ", cols=" << ncols << ", mult=" << mult <<"\n";

    int *rr = scinew int[nrows+1];
    const int nnz = nrows*mult;
    int *cc = scinew int[nnz];
    double *d = scinew double[nnz];

    if (at == Field::EDGE)
    {
      rr[0] = 0;
      int counter = 0;
      typename FSRC::mesh_type::Edge::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      while (itr != eitr)
      {
	mesh->get_nodes(tmparray, *itr);
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
      rr[0] = 0;
      int counter = 0;
      typename FSRC::mesh_type::Face::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      while (itr != eitr)
      {
	mesh->get_nodes(tmparray, *itr);
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
      rr[0] = 0;
      int counter = 0;
      typename FSRC::mesh_type::Cell::iterator itr, eitr;
      mesh->begin(itr);
      mesh->end(eitr);
      while (itr != eitr)
      {
	mesh->get_nodes(tmparray, *itr);
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

    interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);
  }

  fout->copy_properties(fsrc);

  return fout;
}


} // end namespace SCIRun

#endif // ChangeFieldDataAt_h
