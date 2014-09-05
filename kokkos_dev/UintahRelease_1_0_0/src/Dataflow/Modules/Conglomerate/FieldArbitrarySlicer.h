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


//    File   : FieldArbitrarySlicer.h
//    Author : Michael Callahan &&
//             Allen Sanderson
//             SCI Institute
//             University of Utah
//    Date   : March 2006
//
//    Copyright (C) 2006 SCI Group


#if !defined(FieldArbitrarySlicer_h)
#define FieldArbitrarySlicer_h

#include <sci_defs/insight_defs.h>

#include <Core/Containers/Handle.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/ScanlineMesh.h>
//#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Datatypes/StructCurveMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Util/TypeDescription.h>

#ifdef HAVE_INSIGHT
#include "Packages/Insight/Core/Datatypes/ITKLatVolField.h"
#include "Packages/Insight/Core/Datatypes/ITKImageField.h"
#endif

namespace SCIRun {

#ifdef HAVE_INSIGHT
using namespace Insight;
#endif

class FieldArbitrarySlicerAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src, int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ttd);
};

template< class FIELD, class TYPE >
class FieldArbitrarySlicerAlgoT : public FieldArbitrarySlicerAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src, int axis);

};


class FieldArbitrarySlicerWorkAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle& src,
		       FieldHandle& dst,
		       unsigned int index,
		       int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd,
					    const TypeDescription *oftd,
					    bool geomtery_irregular);
};

template< class IFIELD, class OFIELD >
class FieldArbitrarySlicerWorkAlgoT : public FieldArbitrarySlicerWorkAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle& src,
		       FieldHandle& dst,
		       unsigned int index,
		       int axis);
};


template< class FIELD, class TYPE >
FieldHandle
FieldArbitrarySlicerAlgoT<FIELD, TYPE>::execute(FieldHandle& ifield_h, int axis)
{
  FieldHandle ofield_h;

  FIELD *ifield = (FIELD *) ifield_h.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  unsigned int old_min_i, old_min_j, old_min_k;
  unsigned int old_i, old_j, old_k;

  unsigned int new_min_i, new_min_j, new_i, new_j;

  vector<unsigned int> dim;
  imesh->get_dim( dim );

  vector<unsigned int> min;
  imesh->get_min( min );

  //! Get the dimensions of the old field.
  if( dim.size() == 3 ) {
    old_min_i = min[0];           old_i = dim[0];
    old_min_j = min[1];           old_j = dim[1];
    old_min_k = min[2];           old_k = dim[2];

  } else if( dim.size() == 2 ) {
    old_min_i = min[0];           old_i = dim[0];
    old_min_j = min[1];           old_j = dim[1];
    old_min_j = 0;                old_k = 1;
  } else if( dim.size() == 1 ) {
    old_min_i = min[0];           old_i = dim[0];
    old_min_j = 0;                old_j = 1;
    old_min_j = 0;                old_k = 1;
  }

  //! Get the dimensions of the new field.
  if (axis == 0) {
    new_i = old_j;                new_min_i = old_min_j;
    new_j = old_k;                new_min_j = old_min_k;
  } else if (axis == 1) {
    new_i = old_i;                new_min_i = old_min_i;
    new_j = old_k;                new_min_j = old_min_k;
  } else if (axis == 2) {
    new_i = old_i;                new_min_i = old_min_i;
    new_j = old_j;                new_min_j = old_min_j;
  }

  //! Build the correct output field given the input field and and type.
  string mesh_type =
    ifield->get_type_description(Field::MESH_TD_E)->get_name();

  //! 3D LatVol to 2D Image
  if( mesh_type.find("LatVolMesh") != string::npos ) {  
    typedef ImageMesh<QuadBilinearLgn<Point> > IMesh;

    IMesh *omesh = scinew IMesh();

    omesh->copy_properties(imesh.get_rep());

    omesh->set_min_i( new_min_i );
    omesh->set_min_j( new_min_j );
    omesh->set_ni( new_i );
    omesh->set_nj( new_j );

    Field *ofield = 0;
    typedef FData2d<TYPE, IMesh> FData;
    if (ifield->basis_order() == -1) {
      typedef GenericField<IMesh, NoDataBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else if (ifield->basis_order() == 0) {
      typedef GenericField<IMesh, ConstantBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else {
      typedef QuadBilinearLgn<TYPE>              LinBasis;
      typedef GenericField<IMesh, LinBasis, FData> OField;
      ofield = scinew OField(omesh);
    }

    ofield->copy_properties(ifield);
    ofield_h = ofield;

    //! 3D ITKLatVolField to 2D ITKImage
#ifdef HAVE_INSIGHT
  } else if( mesh_type.find("ITKLatVolField") != string::npos ) {
    
    //! These really should be ITKImageFields but the conversion will
    //! not work so for now make the ImageFields instead.
    typedef ImageMesh<QuadBilinearLgn<Point> > IMesh;
    IMesh *omesh = scinew IMesh();
    omesh->copy_properties(imesh.get_rep());

    omesh->set_min_i( new_min_i );
    omesh->set_min_j( new_min_j );
    omesh->set_ni( new_i );
    omesh->set_nj( new_j );
    
    Field *ofield = 0;
    typedef FData2d<TYPE, IMesh> FData;
    if (ifield->basis_order() == -1) {
      typedef GenericField<IMesh, NoDataBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else if (ifield->basis_order() == 0) {
      typedef GenericField<IMesh, ConstantBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else {
      typedef QuadBilinearLgn<TYPE>              LinBasis;
      typedef GenericField<IMesh, LinBasis, FData> OField;
      ofield = scinew OField(omesh);
    }

    ofield->copy_properties(ifield);
    ofield_h = ofield;
#endif
    //! 3D StructHexVol to 2D StructQuadSurf
  } else if( mesh_type.find("StructHexVolMesh") != string::npos ) {

    typedef StructQuadSurfMesh<QuadBilinearLgn<Point> > SQMesh;

    SQMesh *omesh = scinew SQMesh(new_i, new_j);
    omesh->copy_properties(imesh.get_rep());

    Field *ofield = 0;
    typedef FData2d<TYPE, SQMesh> FData;
    if (ifield->basis_order() == -1) {
      typedef GenericField<SQMesh, NoDataBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else if (ifield->basis_order() == 0) {
      typedef GenericField<SQMesh, ConstantBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else {
      typedef QuadBilinearLgn<TYPE>               LinBasis;
      typedef GenericField<SQMesh, LinBasis, FData> OField;
      ofield = scinew OField(omesh);
    }

    ofield->copy_properties(ifield);
    ofield_h = ofield;

    //! 2D Image to 1D Scanline or 
    //! 1D Scanline to 0D Scanline (perhaps it should be pointcloud).
  } else if( mesh_type.find("ImageMesh")    != string::npos || 
	     mesh_type.find("ScanlineMesh") != string::npos) {

    typedef ScanlineMesh<CrvLinearLgn<Point> > SLMesh;

    SLMesh *omesh = scinew SLMesh();
    omesh->copy_properties(imesh.get_rep());

    omesh->set_min_i( new_min_i );
    omesh->set_ni( new_i );

    Field *ofield = 0;
    typedef vector<TYPE> FData;
    if (ifield->basis_order() == -1) {
      typedef GenericField<SLMesh, NoDataBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else if (ifield->basis_order() == 0) {
      typedef GenericField<SLMesh, ConstantBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else {
      typedef CrvLinearLgn<TYPE>                  LinBasis;
      typedef GenericField<SLMesh, LinBasis, FData> OField;
      ofield = scinew OField(omesh);
    }

    ofield->copy_properties(ifield);
    ofield_h = ofield;

    //! 2D ITKImage to 1D Scanline
#ifdef HAVE_INSIGHT
  } else if( mesh_type.find("ITKImageField") != string::npos ) {

    //! These really should be ITKScanlineFields but the conversion will
    //! not work so for now make the ScanlineFields instead.
    typedef ScanlineMesh<CrvLinearLgn<Point> > SLMesh;

    SLMesh *omesh = scinew SLMesh();
    omesh->copy_properties(imesh.get_rep());

    omesh->set_min_i( new_min_i );
    omesh->set_ni( new_i );

    Field *ofield = 0;
    typedef vector<TYPE> FData;
    if (ifield->basis_order() == -1) {
      typedef GenericField<SLMesh, NoDataBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else if (ifield->basis_order() == 0) {
      typedef GenericField<SLMesh, ConstantBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else {
      typedef CrvLinearLgn<TYPE>                  LinBasis;
      typedef GenericField<SLMesh, LinBasis, FData> OField;
      ofield = scinew OField(omesh);
    }

    ofield->copy_properties(ifield);
    ofield_h = ofield;
#endif
    //! 2D StructQuadSurf to 1D StructCurve
  } else if( mesh_type.find("StructQuadSurfMesh") != string::npos ) {
    typedef StructCurveMesh<CrvLinearLgn<Point> > SCMesh;

    SCMesh *omesh = scinew SCMesh(new_i);
    omesh->copy_properties(imesh.get_rep());

    Field *ofield = 0;
    typedef vector<TYPE> FData;
    if (ifield->basis_order() == -1) {
      typedef GenericField<SCMesh, NoDataBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else if (ifield->basis_order() == 0) {
      typedef GenericField<SCMesh, ConstantBasis<TYPE>, FData > OField;
      ofield = scinew OField(omesh);
    } else {
      typedef CrvLinearLgn<TYPE>                  LinBasis;
      typedef GenericField<SCMesh, LinBasis, FData> OField;
      ofield = scinew OField(omesh);
    }

    ofield->copy_properties(ifield);
    ofield_h = ofield;

    // 1D StructCurve to 0D PointCloud
  } else if( mesh_type.find("StructCurveMesh") != string::npos ) {

    typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;
    PCMesh *omesh = scinew PCMesh();
    omesh->copy_properties(imesh.get_rep());

    Field *ofield = 0;
    typedef vector<TYPE> FData;
    typedef GenericField<PCMesh, ConstantBasis<TYPE>, FData > OField;
    OField *ofld = scinew OField(omesh);
    ofield = ofld;
 
    ofield->copy_properties(ifield);
    ofield_h = ofield;

    omesh->resize_nodes( new_i );
    ofld->resize_fdata();
  }

  return ofield_h;
}


template< class IFIELD, class OFIELD >
void
FieldArbitrarySlicerWorkAlgoT<IFIELD, OFIELD>::execute(FieldHandle& ifield_h,
					      FieldHandle& ofield_h,
					      unsigned int index,
					      int axis)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  OFIELD *ofield = (OFIELD *) ofield_h.get_rep();
  typename OFIELD::mesh_handle_type omesh = ofield->get_typed_mesh();

  unsigned int old_i, old_j, old_k;
  unsigned int new_i, new_j;

  vector<unsigned int> dim;
  imesh->get_dim( dim );

  if( dim.size() == 3 ) {
    old_i = dim[0];
    old_j = dim[1];
    old_k = dim[2];
  } else if( dim.size() == 2 ) {
    old_i = dim[0];
    old_j = dim[1];
    old_k = 1;      //! This makes it is possible to slice from 1D to 0D easily.
  } else if( dim.size() == 1 ) {
    old_i = dim[0];
    old_j = 1;
    old_k = 1;      //! This makes it is possible to slice from 1D to 0D easily.
  }

  if (axis == 0) {
    new_i = old_j;
    new_j = old_k;

  } else if (axis == 1) {
    new_i = old_i;
    new_j = old_k;

  } else if (axis == 2) {
    new_i = old_i;
    new_j = old_j;
  }

  typename IFIELD::mesh_type::Node::iterator inodeItr;
  typename OFIELD::mesh_type::Node::iterator onodeItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  Point p;
  typename IFIELD::value_type v;
 
  unsigned int i, j;
  unsigned int it, jt, kt;

#ifndef SET_POINT_DEFINED
  //! For structured geometry we need to set the correct plane.
  string mesh_type =
    ifield->get_type_description(Field::MESH_TD_E)->get_name();

  if( mesh_type.find("LatVolMesh"  ) != string::npos ||
      mesh_type.find("ImageMesh"   ) != string::npos ||
      mesh_type.find("ScanlineMesh") != string::npos )
  {
    Transform trans = imesh->get_transform();
    double offset = 0.0;
    if (axis == 0)
    {
      trans.post_permute(2, 3, 1);
      offset = index / (double)old_i;
    }
    else if (axis == 1)
    {
      trans.post_permute(1, 3, 2);
      offset = index / (double)old_j;
    }
    else
    {
      offset = index / (double)old_k;
    }
    trans.post_translate(Vector(0.0, 0.0, index));

    omesh->transform( trans );
  }
#endif

  imesh->begin( inodeItr );

  //! Slicing along the i axis. In order to get the slice in this
  //! direction only one location can be obtained at a time. So the
  //! iterator must increment to the correct column (i) index, the
  //! location samples, and then incremented to the end of the row.
  if (axis == 0) {                  // node.i_ = index
    for (j=0; j<new_j; j++) {       // node.j_ = i
      for (i=0; i<new_i; i++) {     // node.k_ = j
	
	// Set the iterator to the correct column (i) .
	for (it=0; it<index; it++)
	  ++inodeItr;

	// Get the point and value at this location
#ifdef SET_POINT_DEFINED
	imesh->get_center(p, *inodeItr);
	omesh->set_point(p, *onodeItr);
#endif       
	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++onodeItr;

	// Move to the end of the row.
	for (;it<old_i; it++)
	  ++inodeItr;
      }
    }
    //! Slicing along the j axis. In order to get the slice in this
    //! direction a complete row can be obtained at a time. So the
    //! iterator must increment to the correct row (j) index, the
    //! location samples, and then incremented to the end of the slice.
  } else if(axis == 1) {              // node.i_ = i
    for (j=0; j < new_j; j++) {       // node.j_ = index
                                      // node.k_ = j
      //! Set the iterator to the correct row (j).
      for (jt=0; jt<index; jt++)
	for (it=0; it<old_i; it++) 
	  ++inodeItr;

      //! Get all of the points and values along this row.
      for (i=0; i<new_i; i++) {

	//! Get the point and value at this location
#ifdef SET_POINT_DEFINED
	imesh->get_center(p, *inodeItr);
	omesh->set_point(p, *onodeItr);
#endif
	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++inodeItr;
	++onodeItr;
      }

      //! Just got one full row line so increment.
      jt++;

      //! Move to the end of all the rows..
      for (; jt<old_j; jt++)
	for (it=0; it<old_i; it++)
	  ++inodeItr;
    }

    //! Slicing along the k axis.
  } else if(axis == 2) {
    //! Set the iterator to the correct k slice.
    for (kt=0; kt<index; kt++)        // node.i_ = i
      for (j=0; j<old_j; j++)         // node.j_ = j
	for (i=0; i<old_i; i++)       // node.k_ = index
	  ++inodeItr;

    //! Get all of the points and values along this slice.
    for (j=0; j < new_j; j++) {
      for (i=0; i < new_i; i++) {

	//! Get the point and value at this location
#ifdef SET_POINT_DEFINED
	imesh->get_center(p, *inodeItr);
	omesh->set_point(p, *onodeItr);
#endif
	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++inodeItr;
	++onodeItr;
      }
    }
  }
}

} //! end namespace SCIRun

#endif //! FieldArbitrarySlicer_h
