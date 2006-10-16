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


//    File   : MoveElemToNode.h
//    Author : Michael Callahan
//    Date   : September 2003

#if !defined(MoveElemToNode_h)
#define MoveElemToNode_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Geometry/Transform.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {


class GuiInterface;

class MoveElemToNodeAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter,
			      FieldHandle fieldh) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
                                            const string &fdst,
					    const string &ext);
};



template <class FSRC, class FDST>
class MoveElemToNodeAlgoLat : public MoveElemToNodeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);

};


template <class FSRC, class FDST>
FieldHandle
MoveElemToNodeAlgoLat<FSRC, FDST>::execute(ProgressReporter *reporter,
                                           FieldHandle fieldh)
{
  FSRC *ifield = dynamic_cast<FSRC*>(fieldh.get_rep());
  typename FSRC::mesh_type *imesh =
    dynamic_cast<typename FSRC::mesh_type *>(fieldh->mesh().get_rep());

  const int ni = imesh->get_ni();
  const int nj = imesh->get_nj();
  const int nk = imesh->get_nk();

  const double ioff = (1.0 - ((ni-2.0) / (ni-1.0))) * 0.5;
  const double joff = (1.0 - ((nj-2.0) / (nj-1.0))) * 0.5;
  const double koff = (1.0 - ((nk-2.0) / (nk-1.0))) * 0.5;

  const Point minp(ioff, joff, koff);
  const Point maxp(1.0-ioff, 1.0-joff, 1.0-koff);

  typename FDST::mesh_type *omesh =
    scinew typename FDST::mesh_type(ni-1, nj-1, nk-1, minp, maxp);

  Transform trans;
  imesh->get_canonical_transform(trans);
  omesh->transform(trans);

  FDST *ofield = scinew FDST(omesh);

  // Copy data from ifield to ofield.
  typename FSRC::mesh_type::Elem::iterator iter, eiter;
  imesh->begin(iter);
  imesh->end(eiter);
  while (iter != eiter)
  {
    typename FSRC::value_type v;
    ifield->value(v, *iter);

    typename FDST::mesh_type::Node::index_type oi(omesh,
                                                  (*iter).i_,
                                                  (*iter).j_,
                                                  (*iter).k_);
    ofield->set_value(v, oi);

    ++iter;
  }

  return ofield;
}


template <class FSRC, class FDST>
class MoveElemToNodeAlgoSHex : public MoveElemToNodeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);

};


template <class FSRC, class FDST>
FieldHandle
MoveElemToNodeAlgoSHex<FSRC, FDST>::execute(ProgressReporter *reporter,
                                            FieldHandle fieldh)
{
  FSRC *ifield = dynamic_cast<FSRC*>(fieldh.get_rep());
  typename FSRC::mesh_type *imesh =
    dynamic_cast<typename FSRC::mesh_type *>(fieldh->mesh().get_rep());

  const int ni = imesh->get_ni()-1;
  const int nj = imesh->get_nj()-1;
  const int nk = imesh->get_nk()-1;

  typename FDST::mesh_type *omesh =
    scinew typename FDST::mesh_type(ni, nj, nk);

  int i, j, k;
  for (i = 0; i < ni; i++)
  {
    for (j = 0; j < nj; j++)
    {
      for (k = 0; k < nk; k++)
      {
	Point p;
	typename FSRC::mesh_type::Elem::index_type ii(imesh, i, j, k);
	imesh->get_center(p, ii);

	typename FDST::mesh_type::Node::index_type oi(omesh, i, j, k);
	omesh->set_point(p, oi);
      }
    }
  }

  FDST *ofield = scinew FDST(omesh);

  // Copy data from ifield to ofield.
  typename FSRC::mesh_type::Elem::iterator iter, eiter;
  imesh->begin(iter);
  imesh->end(eiter);
  while (iter != eiter)
  {
    typename FSRC::value_type v;
    ifield->value(v, *iter);

    typename FDST::mesh_type::Node::index_type oi(omesh,
                                                  (*iter).i_,
                                                  (*iter).j_,
                                                  (*iter).k_);

    ofield->set_value(v, oi);

    ++iter;
  }

  return ofield;
}



template <class FSRC, class FDST>
class MoveElemToNodeAlgoImg : public MoveElemToNodeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);

};


template <class FSRC, class FDST>
FieldHandle
MoveElemToNodeAlgoImg<FSRC, FDST>::execute(ProgressReporter *reporter,
                                           FieldHandle fieldh)
{
  FSRC *ifield = dynamic_cast<FSRC*>(fieldh.get_rep());
  typename FSRC::mesh_type *imesh =
    dynamic_cast<typename FSRC::mesh_type *>(fieldh->mesh().get_rep());

  const int ni = imesh->get_ni();
  const int nj = imesh->get_nj();

  const double ioff = (1.0 - ((ni-2.0) / (ni-1.0))) * 0.5;
  const double joff = (1.0 - ((nj-2.0) / (nj-1.0))) * 0.5;

  const Point minp(ioff, joff, 0.0);
  const Point maxp(1.0-ioff, 1.0-joff, 1.0);

  typename FDST::mesh_type *omesh =
    scinew typename FDST::mesh_type(ni-1, nj-1, minp, maxp);

  Transform trans;
  imesh->get_canonical_transform(trans);
  omesh->transform(trans);

  FDST *ofield = scinew FDST(omesh);

  // Copy data from ifield to ofield.
  typename FSRC::mesh_type::Elem::iterator iter, eiter;
  imesh->begin(iter);
  imesh->end(eiter);
  while (iter != eiter)
  {
    typename FSRC::value_type v;
    ifield->value(v, *iter);

    typename FDST::mesh_type::Node::index_type oi(omesh,
                                                  (*iter).i_,
                                                  (*iter).j_);

    ofield->set_value(v, oi);

    ++iter;
  }

  return ofield;
}



template <class FSRC, class FDST>
class MoveElemToNodeAlgoSQuad : public MoveElemToNodeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);

};


template <class FSRC, class FDST>
FieldHandle
MoveElemToNodeAlgoSQuad<FSRC, FDST>::execute(ProgressReporter *reporter,
                                             FieldHandle fieldh)
{
  FSRC *ifield = dynamic_cast<FSRC*>(fieldh.get_rep());
  typename FSRC::mesh_type *imesh =
    dynamic_cast<typename FSRC::mesh_type *>(fieldh->mesh().get_rep());

  const int ni = imesh->get_ni()-1;
  const int nj = imesh->get_nj()-1;

  typename FDST::mesh_type *omesh =
    scinew typename FDST::mesh_type(ni, nj);

  int i, j;
  for (i = 0; i < ni; i++)
  {
    for (j = 0; j < nj; j++)
    {
      Point p;
      typename FSRC::mesh_type::Elem::index_type ii(imesh, i, j);
      imesh->get_center(p, ii);

      typename FDST::mesh_type::Node::index_type oi(omesh, i, j);
      omesh->set_point(p, oi);
    }
  }

  FDST *ofield = scinew FDST(omesh);

  // Copy data from ifield to ofield.
  typename FSRC::mesh_type::Elem::iterator iter, eiter;
  imesh->begin(iter);
  imesh->end(eiter);
  while (iter != eiter)
  {
    typename FSRC::value_type v;
    ifield->value(v, *iter);

    typename FDST::mesh_type::Node::index_type oi(omesh,
                                                  (*iter).i_,
                                                  (*iter).j_);

    ofield->set_value(v, oi);

    ++iter;
  }

  return ofield;
}



} // end namespace SCIRun

#endif // MoveElemToNode_h
