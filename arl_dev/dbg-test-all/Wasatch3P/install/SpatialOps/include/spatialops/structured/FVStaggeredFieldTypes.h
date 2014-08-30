/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef FVStaggeredTypes_h
#define FVStaggeredTypes_h

#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/FVStaggeredLocationTypes.h>
#include <spatialops/structured/SpatialField.h>


/**
 *  \file FVStaggeredFieldTypes.h
 */

namespace SpatialOps{

  /**
     \page fields-meshes Fields and Meshes

     \tableofcontents
     \sa The \ref fieldtypes module, which provides additional information on the types discussed here.
     \sa The example discussing \ref example-field-types
     \sa The example discussing \ref example-field-type-inference

     \section field-types Field types on meshes

     The following table enumerates the field types defined in SpatialOps associated with each mesh.
     -         |  Scalar Mesh  |  X-Mesh       |  Y-Mesh       |  Z-Mesh
     --------- | ------------- | ------------- | ------------- | -------------
     Volume    | `  SVolField` | `  XVolField` | `  YVolField` | `  ZVolField`
     X-Surface | `SSurfXField` | `XSurfXField` | `YSurfXField` | `ZSurfXField`
     Y-Surface | `SSurfYField` | `XSurfYField` | `YSurfYField` | `ZSurfYField`
     Z-Surface | `SSurfZField` | `XSurfZField` | `YSurfZField` | `ZSurfZField`


     \section svol The Scalar (non-staggered) Volume

     The following image illustrates a scalar mesh in 2D with the associated
     field locations.  Filled circles indicate volume centroids for cells within
     the domain while empty circles indicate ghost volume locations.
     \image html SVolFields_2D.jpg "Field types in 2D with their locations on a scalar (non-staggered) mesh"

     \subsection svol-observations Key observations on the scalar volume
      - This figure illustrates one layer of ghost cells (depicted in shaded grey).
      - Face fields on the (-) side face are associated with the cell volume for
        indexing purposes.
      - When a domain boundary is present, the surface fields have an extra
        storage location.  For a mesh of size `(nx,ny,nz)` with (+) boundaries
        in each direction,
        - `SVolField` is dimension `(nx,ny,nz)`
        - `SSurfXField` is dimension `(nx+1,ny,nz)`
        - `SSurfYField` is dimension `(nx,ny+1,nz)`
        - `SSurfZField` is dimension `(nx,ny,nz+1)`
      - The figure above depicts a situation where there is no (+) boundary present.


     \section xvol The X-Staggered Volume

     The following image illustrates a scalar mesh in 2D with the associated
     field locations.  Filled squares indicate x-volume centroids located within
     while the domain while empty squares indicate ghost values.
     \image html SVol_XVol.jpg "The scalar and x-staggered volume"

     \subsection xvol-observations Key observations on the x-staggered volume
       - For a uniform mesh (depicted here), the `XVolField` is colocated with
         the `SSurfXField`
       - Just as with \ref svol, face fields on the (-) side are associated with
         the cell volume for indexing purposes.
       - When a domain boundary is present, the surface fields have an extra
         storage location.  For a mesh of size `(nx,ny,nz)`
         - `XVolField` is dimension `(nx,ny,nz)`
         - `XSurfXField` is dimension `(nx+1,ny,nz)`
         - `XSurfYField` is dimension `(nx,ny+1,nz)`
         - `XSurfZField` is dimension `(nx,ny,nz+1)`
       - Note that even though `XVolField` is colocated with `SSurfXField`, its
         storage behavior is different with respect to (+) side boundaries.

     \section yvol The Y-Staggered Volume

     The following image illustrates a scalar mesh in 2D with the associated
     field locations. Filled triangles indicate y-volume cell centroids located
     within the domain while empty triangles indicate values in ghost cells.
     \image html SVol_YVol.jpg "The scalar and x-staggered volume"

     \subsection yvol-observations Key observations on the y-staggered volume
       - For a uniform mesh (depicted here), the `YVolField` is colocated with
         the `SSurfYField`
       - Just as with \ref svol, face fields on the (-) side are associated with
         the cell volume for indexing purposes.
       - When a domain boundary is present, the surface fields have an extra
         storage location.  For a mesh of size `(nx,ny,nz)`
         - `YVolField` is dimension `(nx,ny,nz)`
         - `YSurfXField` is dimension `(nx+1,ny,nz)`
         - `YSurfYField` is dimension `(nx,ny+1,nz)`
         - `YSurfZField` is dimension `(nx,ny,nz+1)`
       - Note that even though `YVolField` is colocated with `SSurfYField`, its
         storage behavior is different with respect to (+) side boundaries.

     \section zvol The Z-Staggered Volume
     The z-staggered volume behaves in an analogous way to the \ref xvol and \ref yvol discussions above.

     \section field-type-inference Field type inference
     There are two key tools to aid in type inference:
      -# Given a volume field type, the associated face field types may be obtained using the FaceTypes struct.
      -# Given a face field type, the associated volume field type may be obtained using the VolType struct.
     \sa The \ref example-field-type-inference example
   */


  /**
   *  \addtogroup fieldtypes
   *  @{
   *  \sa \ref fields-meshes
   *
   *  \typedef typedef SVolField;
   *  \brief defines a volume field on the scalar volume.
   *
   *  \typedef typedef SSurfXField;
   *  \brief defines a x-surface field on the scalar volume
   *
   *  \typedef typedef SSurfYField;
   *  \brief defines a y-surface field on the scalar volume
   *
   *  \typedef typedef SSurfZField;
   *  \brief defines a z-surface field on the scalar volume
   */
  typedef SpatialField< SVol   > SVolField;
  typedef SpatialField< SSurfX > SSurfXField;
  typedef SpatialField< SSurfY > SSurfYField;
  typedef SpatialField< SSurfZ > SSurfZField;


  /**
   *  \typedef typedef XVolField;
   *  \brief defines a volume field on the x-staggered volume
   *
   *  \typedef typedef XSurfXField;
   *  \brief defines a x-surface field on the x-staggered volume
   *
   *  \typedef typedef XSurfYField;
   *  \brief defines a y-surface field on the x-staggered volume
   *
   *  \typedef typedef XSurfZField;
   *  \brief defines a z-surface field on the x-staggered volume
   */
  typedef SpatialField< XVol   > XVolField;
  typedef SpatialField< XSurfX > XSurfXField;
  typedef SpatialField< XSurfY > XSurfYField;
  typedef SpatialField< XSurfZ > XSurfZField;


  /**
   *  \typedef typedef YVolField;
   *  \brief defines a volume field on the y-staggered volume
   *
   *  \typedef typedef YSurfXField;
   *  \brief defines a x-surface field on the y-staggered volume
   *
   *  \typedef typedef YSurfYField;
   *  \brief defines a y-surface field on the y-staggered volume
   *
   *  \typedef typedef YSurfZField;
   *  \brief defines a z-surface field on the y-staggered volume
   */
  typedef SpatialField< YVol   > YVolField;
  typedef SpatialField< YSurfX > YSurfXField;
  typedef SpatialField< YSurfY > YSurfYField;
  typedef SpatialField< YSurfZ > YSurfZField;


  /**
   *  \typedef typedef ZVolField;
   *  \brief defines a volume field on the z-staggered volume
   *
   *  \typedef typedef ZSurfXField;
   *  \brief defines a x-surface field on the z-staggered volume
   *
   *  \typedef typedef ZSurfYField;
   *  \brief defines a y-surface field on the z-staggered volume
   *
   *  \typedef typedef ZSurfZField;
   *  \brief defines a z-surface field on the z-staggered volume
   */
  typedef SpatialField< ZVol   > ZVolField;
  typedef SpatialField< ZSurfX > ZSurfXField;
  typedef SpatialField< ZSurfY > ZSurfYField;
  typedef SpatialField< ZSurfZ > ZSurfZField;


  /**
   *  \typedef typedef SpatialField< SingleValue > SingleValueField;
   *  \brief defines a single value field
   *
   *  Warning: SingleValueFields should ONLY be built with MemoryWindows of size 1x1x1!
   */
  typedef SpatialField< SingleValue > SingleValueField;

  /**
   *  \struct FaceTypes
   *  \brief Define Face field types in terms of a cell field type.
   *
   *  Class template specializations exist for the following field types:
   *   - `SVolField`
   *   - `XVolField`
   *   - `YVolField`
   *   - `ZVolField`
   *
   *  Specializations of this struct define the following typedefs:
   *   - `XFace` - the type of the field on the x-face
   *   - `YFace` - the type of the field on the y-face
   *   - `ZFace` - the type of the field on the z-face
   *
   *  Example usage:
   *  \code{.cpp}
   *  typedef FaceTypes< CellT >::XFace XFaceT;
   *  typedef FaceTypes< CellT >::YFace YFaceT;
   *  typedef FaceTypes< CellT >::ZFace ZFaceT;
   *  \endcode
   *
   *  See also \ref VolType.
   */
  template< typename CellFieldT > struct FaceTypes;

  template<> struct FaceTypes<SVolField>
  {
    typedef SSurfXField XFace;
    typedef SSurfYField YFace;
    typedef SSurfZField ZFace;
  };

  template<> struct FaceTypes<XVolField>
  {
    typedef XSurfXField XFace;
    typedef XSurfYField YFace;
    typedef XSurfZField ZFace;
  };

  template<> struct FaceTypes<YVolField>
  {
    typedef YSurfXField XFace;
    typedef YSurfYField YFace;
    typedef YSurfZField ZFace;
  };

  template<> struct FaceTypes<ZVolField>
  {
    typedef ZSurfXField XFace;
    typedef ZSurfYField YFace;
    typedef ZSurfZField ZFace;
  };


  /**
   *  \struct VolType
   *  \brief Define face field types in terms of a volume field type.
   *
   *  Class template specializations exist for the following field types:
   *   - `SSurfXField`
   *   - `SSurfYField`
   *   - `SSurfZField`
   *   - `XSurfXField`
   *   - `XSurfYField`
   *   - `XSurfZField`
   *   - `YSurfXField`
   *   - `YSurfYField`
   *   - `YSurfZField`
   *   - `ZSurfXField`
   *   - `ZSurfYField`
   *   - `ZSurfZField`
   *
   *  Example usage:
   *  \code{.cpp}
   *  typedef VolType< FaceT       >::VolField FieldT;
   *  typedef VolType< SSurfZField >::VolField FieldT;
   *  \endcode
   *
   *  See also \ref FaceTypes.
   */
  template<typename FaceT> struct VolType;

  template<> struct VolType<SSurfXField>{ typedef SVolField VolField; };
  template<> struct VolType<SSurfYField>{ typedef SVolField VolField; };
  template<> struct VolType<SSurfZField>{ typedef SVolField VolField; };

  template<> struct VolType<XSurfXField>{ typedef XVolField VolField; };
  template<> struct VolType<XSurfYField>{ typedef XVolField VolField; };
  template<> struct VolType<XSurfZField>{ typedef XVolField VolField; };

  template<> struct VolType<YSurfXField>{ typedef YVolField VolField; };
  template<> struct VolType<YSurfYField>{ typedef YVolField VolField; };
  template<> struct VolType<YSurfZField>{ typedef YVolField VolField; };

  template<> struct VolType<ZSurfXField>{ typedef ZVolField VolField; };
  template<> struct VolType<ZSurfYField>{ typedef ZVolField VolField; };
  template<> struct VolType<ZSurfZField>{ typedef ZVolField VolField; };

/**
 *  @} // fieldtypes group
 */

}// namespace SpatialOps


#endif
