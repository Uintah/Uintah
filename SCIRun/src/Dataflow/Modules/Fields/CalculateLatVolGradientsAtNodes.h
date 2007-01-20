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


/*
 *  CalculateLatVolGradientsAtNodes.h:
 *
 *  Written by:
 *   Michael Callahan
 *   School of Computering
 *   University of Utah
 *   June 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#if !defined(CalculateLatVolGradientsAtNodes_h)
#define CalculateLatVolGradientsAtNodes_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>

namespace SCIRun {

class CalculateLatVolGradientsAtNodesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd);
};


template< class IFIELD >
class CalculateLatVolGradientsAtNodesAlgoT : public CalculateLatVolGradientsAtNodesAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src);
};


template< class IFIELD >
FieldHandle
CalculateLatVolGradientsAtNodesAlgoT<IFIELD>::execute(FieldHandle& field_h)
{
  IFIELD *ifield = (IFIELD *) field_h.get_rep();
  typedef LatVolMesh<HexTrilinearLgn<Point> >  LVMesh;
  typedef HexTrilinearLgn<Vector>              DatBasis;
  typedef GenericField<LVMesh, DatBasis,  FData3d<Vector, LVMesh> > LVField;
  LVMesh::handle_type imesh = ifield->get_typed_mesh();
    
  LVField *ofield = scinew LVField(imesh);

  LVMesh::Node::iterator itr, end;

  imesh->begin( itr );
  imesh->end( end );

  LVMesh::Node::size_type size;
  imesh->size(size);

  const unsigned int ni = size.i_-1;
  const unsigned int nj = size.j_-1;
  const unsigned int nk = size.k_-1;

  const Transform &transform = imesh->get_transform();

  while (itr != end)
    {
      // Get all of the adjacent indices.  Clone boundary.
      LVMesh::Node::index_type ix0(*itr), ix1(*itr);
      LVMesh::Node::index_type iy0(*itr), iy1(*itr);
      LVMesh::Node::index_type iz0(*itr), iz1(*itr);

      double xscale, yscale, zscale;
      xscale=yscale=zscale=0.5;

      if (ix0.i_ > 0)  { ix0.i_--; } else { xscale=1.0; }
      if (ix1.i_ < ni) { ix1.i_++; } else { xscale=1.0; }
      if (iy0.j_ > 0)  { iy0.j_--; } else { yscale=1.0; }
      if (iy1.j_ < nj) { iy1.j_++; } else { yscale=1.0; }
      if (iz0.k_ > 0)  { iz0.k_--; } else { zscale=1.0; }
      if (iz1.k_ < nk) { iz1.k_++; } else { zscale=1.0; }

      // Get all of the adjacent values.
      typename IFIELD::value_type x0, x1, y0, y1, z0, z1;

      ifield->value(x0, ix0);
      ifield->value(x1, ix1);
      ifield->value(y0, iy0);
      ifield->value(y1, iy1);
      ifield->value(z0, iz0);
      ifield->value(z1, iz1);

      // Compute gradient.
      const Vector g((x1 - x0)*xscale, (y1 - y0)*yscale, (z1 - z0)*zscale);

      // Transform gradient to world space.
      const Vector gradient = transform.project_normal(g);

      ofield->set_value(gradient, *itr);
      ++itr;
    }

  ofield->freeze();

  return FieldHandle( ofield );
}

} // end namespace SCIRun

#endif // CalculateLatVolGradientsAtNodes_h
