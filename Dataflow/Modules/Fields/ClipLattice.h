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

//    File   : ClipLattice.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipLattice_h)
#define ClipLattice_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Clipper.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {

class ClipLatticeAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle fieldh,
			      const Point &a, const Point &b) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class ClipLatticeAlgoT : public ClipLatticeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle fieldh,
			      const Point &a, const Point &b);
};


template <class FIELD>
FieldHandle
ClipLatticeAlgoT<FIELD>::execute(FieldHandle fieldh,
				 const Point &a, const Point &b)
{
  FIELD *lv = dynamic_cast<FIELD *>(fieldh.get_rep());
  LatVolMesh *omesh = lv->get_typed_mesh().get_rep();

  LatVolMesh::Node::index_type ns, ne;
  omesh->locate(ns, a);
  omesh->locate(ne, b);

  const int onx = omesh->get_ni();
  const int ony = omesh->get_nj();
  const int onz = omesh->get_nk();

  int si = (int)(ns.i_);
  int sj = (int)(ns.j_);
  int sk = (int)(ns.k_);
  int ei = (int)(ne.i_);
  int ej = (int)(ne.j_);
  int ek = (int)(ne.k_);

  if (si < 0) { si = 0; } 
  if (sj < 0) { sj = 0; } 
  if (sk < 0) { sk = 0; } 
  if (ei < 0) { ei = 0; } 
  if (ej < 0) { ej = 0; } 
  if (ek < 0) { ek = 0; } 

  if (si >= onx) { si = onx - 1; } 
  if (sj >= ony) { sj = ony - 1; } 
  if (sk >= onz) { sk = onz - 1; } 
  if (ei >= onx) { ei = onx - 1; } 
  if (ej >= ony) { ej = ony - 1; } 
  if (ek >= onz) { ek = onz - 1; } 

  LatVolMesh::Node::index_type s, e;

  if (si < ei) { s.i_ = si; e.i_ = ei; }
  else         { s.i_ = ei; e.i_ = si; }
  if (sj < ej) { s.j_ = sj; e.j_ = ej; }
  else         { s.j_ = ej; e.j_ = sj; }
  if (sk < ek) { s.k_ = sk; e.k_ = ek; }
  else         { s.k_ = ek; e.k_ = sk; }

  const int nx = e.i_ - s.i_ + 1;
  const int ny = e.j_ - s.j_ + 1;
  const int nz = e.k_ - s.k_ + 1;

  if (nx < 2 || ny < 2 || nz < 2) return 0;

  Point bmin(0.0, 0.0, 0.0);
  Point bmax(1.0, 1.0, 1.0);
  LatVolMesh *mesh = scinew LatVolMesh(nx, ny, nz, bmin, bmax);

  Transform trans = omesh->get_transform();
  trans.post_translate(Vector(s.i_, s.j_, s.k_));

  mesh->get_transform().load_identity();
  mesh->transform(trans);

  FIELD *fld = scinew FIELD(mesh, lv->data_at());
  *(PropertyManager *)fld = *(PropertyManager *)lv;

  if (lv->data_at() == Field::NODE)
  {
    LatVolMesh::Node::iterator si, ei;
    mesh->begin(si); mesh->end(ei);

    while (si != ei)
    {
      LatVolMesh::Node::index_type idx = *si;
      idx.i_ += s.i_;
      idx.j_ += s.j_;
      idx.k_ += s.k_;

      typename FIELD::value_type val;
      lv->value(val, idx);
      fld->set_value(val, *si);
    
      ++si;
    }
  }
  else if (lv->data_at() == Field::CELL)
  {
    LatVolMesh::Cell::iterator si, ei;
    mesh->begin(si); mesh->end(ei);

    while (si != ei)
    {
      LatVolMesh::Cell::index_type idx = *si;
      idx.i_ += s.i_;
      idx.j_ += s.j_;
      idx.k_ += s.k_;

      typename FIELD::value_type val;
      lv->value(val, idx);
      fld->set_value(val, *si);
    
      ++si;
    }
  }


  return fld;
}


} // end namespace SCIRun

#endif // ClipLattice_h
