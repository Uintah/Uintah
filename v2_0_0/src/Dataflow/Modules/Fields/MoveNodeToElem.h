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

//    File   : ClipField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipField_h)
#define ClipField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Geometry/Transform.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {


class GuiInterface;

class MoveNodeToElemAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter,
			      FieldHandle fieldh) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);
};



template <class FIELD>
class MoveNodeToElemAlgoLat : public MoveNodeToElemAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);

};


template <class FIELD>
FieldHandle
MoveNodeToElemAlgoLat<FIELD>::execute(ProgressReporter *mod,
				      FieldHandle fieldh)
{
  FIELD *ifield = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *imesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());

  const int ni = imesh->get_ni();
  const int nj = imesh->get_nj();
  const int nk = imesh->get_nk();

  const double ioff = (1.0 - (ni / (ni-1.0))) * 0.5;
  const double joff = (1.0 - (nj / (nj-1.0))) * 0.5;
  const double koff = (1.0 - (nk / (nk-1.0))) * 0.5;

  const Point minp(ioff, joff, koff);
  const Point maxp(1.0-ioff, 1.0-joff, 1.0-koff);

  typename FIELD::mesh_type *omesh =
    scinew typename FIELD::mesh_type(ni+1, nj+1, nk+1, minp, maxp);

  Transform trans;
  imesh->get_canonical_transform(trans);
  omesh->transform(trans);

  FIELD *ofield = scinew FIELD(omesh, Field::CELL);

  // Copy data from ifield to ofield.
  typename FIELD::mesh_type::Node::iterator iter, eiter;
  imesh->begin(iter);
  imesh->end(eiter);
  while (iter != eiter)
  {
    typename FIELD::value_type v;
    ifield->value(v, *iter);

    typename FIELD::mesh_type::Elem::index_type oi(omesh,
						   (*iter).i_,
						   (*iter).j_,
						   (*iter).k_);

    ofield->set_value(v, oi);

    ++iter;
  }

  return ofield;
}


template <class FIELD>
class MoveNodeToElemAlgoImg : public MoveNodeToElemAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);

};


template <class FIELD>
FieldHandle
MoveNodeToElemAlgoImg<FIELD>::execute(ProgressReporter *mod,
				      FieldHandle fieldh)
{
  FIELD *ifield = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *imesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());

  const int ni = imesh->get_ni();
  const int nj = imesh->get_nj();

  const double ioff = (1.0 - (ni / (ni-1.0))) * 0.5;
  const double joff = (1.0 - (nj / (nj-1.0))) * 0.5;

  const Point minp(ioff, joff, 0.0);
  const Point maxp(1.0-ioff, 1.0-joff, 1.0);

  typename FIELD::mesh_type *omesh =
    scinew typename FIELD::mesh_type(ni+1, nj+1, minp, maxp);

  Transform trans;
  imesh->get_canonical_transform(trans);
  omesh->transform(trans);

  FIELD *ofield = scinew FIELD(omesh, Field::FACE);

  // Copy data from ifield to ofield.
  typename FIELD::mesh_type::Node::iterator iter, eiter;
  imesh->begin(iter);
  imesh->end(eiter);
  while (iter != eiter)
  {
    typename FIELD::value_type v;
    ifield->value(v, *iter);

    typename FIELD::mesh_type::Elem::index_type oi(omesh,
						   (*iter).i_,
						   (*iter).j_);

    ofield->set_value(v, oi);

    ++iter;
  }

  return ofield;
}



} // end namespace SCIRun

#endif // ClipField_h
