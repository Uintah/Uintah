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

//    File   : CastMLVtoHV.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(CastMLVtoHV_h)
#define CastMLVtoHV_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/HexVolField.h>

namespace SCIRun {

class CastMLVtoHVAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle fsrc,
			      Field::data_location loc) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc,
					    const TypeDescription *ldst);
};


template <class FSRC, class LSRC, class FDST, class LDST>
class CastMLVtoHVAlgoT : public CastMLVtoHVAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle fsrc, Field::data_location loc);
};


template <class FSRC, class LSRC, class FDST, class LDST>
FieldHandle
CastMLVtoHVAlgoT<FSRC, LSRC, FDST, LDST>::execute(FieldHandle lv_h,
						  Field::data_location loc)
{
  FSRC *lv = dynamic_cast<FSRC*>(lv_h.get_rep());
  HexVolMeshHandle hvm = scinew HexVolMesh;

  LatVolMeshHandle lvm = lv->get_typed_mesh();

  // Fill in the nodes and connectivities
  BBox bbox = lvm->get_bounding_box();
  Point min = bbox.min();
  Vector diag = bbox.diagonal();
  const int nx = lvm->get_ni();
  const int ny = lvm->get_nj();
  const int nz = lvm->get_nk();
  const double dx = diag.x()/(nx-1);
  const double dy = diag.y()/(ny-1);
  const double dz = diag.z()/(nz-1);

  int i, j, k;
  int ii, jj, kk;
  Array3<int> connectedNodes(nz, ny, nx);
  connectedNodes.initialize(0);
  for (k=0; k<nz-1; k++)
    for (j=0; j<ny-1; j++)
      for (i=0; i<nx-1; i++) {
	int valid=1;
	for (ii=0; ii<2; ii++)
	  for (jj=0; jj<2; jj++)
	    for (kk=0; kk<2; kk++)
	      if (!lv->mask()(k+kk,j+jj,i+ii)) valid=0; // NODE only?
	if (valid)
	  for (ii=0; ii<2; ii++)
	    for (jj=0; jj<2; jj++)
	      for (kk=0; kk<2; kk++)
		connectedNodes(k+kk,j+jj,i+ii)=1;
      }

  Array3<int> nodeMap(nz, ny, nx);
  nodeMap.initialize(-1);
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++)
	if (connectedNodes(k,j,i))
	  nodeMap(k,j,i) = hvm->add_point(min + Vector(dx*i, dy*j, dz*k));

  for (k=0; k<nz-1; k++)
  {
    for (j=0; j<ny-1; j++)
    {
      for (i=0; i<nx-1; i++)
      {
	HexVolMesh::Node::index_type n000, n001, n010, n011;
	HexVolMesh::Node::index_type n100, n101, n110, n111;
	if ((n000 = nodeMap(k  , j  , i  )) == -1) continue;
	if ((n001 = nodeMap(k  , j  , i+1)) == -1) continue;
	if ((n010 = nodeMap(k  , j+1, i  )) == -1) continue;
	if ((n011 = nodeMap(k  , j+1, i+1)) == -1) continue;
	if ((n100 = nodeMap(k+1, j  , i  )) == -1) continue;
	if ((n101 = nodeMap(k+1, j  , i+1)) == -1) continue;
	if ((n110 = nodeMap(k+1, j+1, i  )) == -1) continue;
	if ((n111 = nodeMap(k+1, j+1, i+1)) == -1) continue;
	hvm->add_hex(n000, n001, n011, n010, n100, n101, n111, n110);
      }
    }
  }

  FDST *hv = scinew FDST(hvm, loc);

  typename LDST::iterator bi, ei;
  hvm->begin(bi);
  hvm->end(ei);

  while (bi != ei)
  {
    Point p;
    hvm->get_center(p, *bi);
    typename LSRC::index_type idx;
    lv->get_typed_mesh()->locate(idx, p);
    typename FSRC::value_type val;
    lv->value(val, idx);
    hv->set_value(val, *bi);

    ++bi;
  }

  return hv;
}


} // end namespace SCIRun

#endif // CastMLVtoHV_h
