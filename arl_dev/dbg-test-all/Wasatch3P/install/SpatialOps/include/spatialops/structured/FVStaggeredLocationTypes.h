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

#ifndef FVStaggeredLocationTypes_h
#define FVStaggeredLocationTypes_h

#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/IndexTriplet.h>


/**
 *  \file FVStaggeredLocationTypes.h
 *
 *  \addtogroup fieldtypes
 *  @{
 *
 */

namespace SpatialOps{


  // FaceDir: The direction relative to its volume field that this field is staggered.
  //
  // Offset : The offset of this field relative to a scalar volume center.
  //
  // BCExtra: If a physical BC is present in the given direction, this
  //          typedef provides a way to determine the number of extra
  //          points in this field relative to a scalar volume field.

  /**
   *  \struct SVol
   *  \brief Type traits for a scalar volume field
   *
   *  \struct SSurfX
   *  \brief Type traits for an x-surface field on a scalar volume
   *
   *  \struct SSurfY
   *  \brief Type traits for an y-surface field on a scalar volume
   *
   *  \struct SSurfZ
   *  \brief Type traits for an z-surface field on a scalar volume
   */
  struct SVol  { typedef NODIR FaceDir;  typedef IndexTriplet< 0, 0, 0> Offset;  typedef IndexTriplet<0,0,0>  BCExtra; };
  struct SSurfX{ typedef XDIR  FaceDir;  typedef IndexTriplet<-1, 0, 0> Offset;  typedef IndexTriplet<1,0,0>  BCExtra; };
  struct SSurfY{ typedef YDIR  FaceDir;  typedef IndexTriplet< 0,-1, 0> Offset;  typedef IndexTriplet<0,1,0>  BCExtra; };
  struct SSurfZ{ typedef ZDIR  FaceDir;  typedef IndexTriplet< 0, 0,-1> Offset;  typedef IndexTriplet<0,0,1>  BCExtra; };

  /**
   *  \struct XVol
   *  \brief Type traits for a x-staggered volume field
   *
   *  \struct XSurfX
   *  \brief Type traits for an x-surface field on a x-staggered volume
   *
   *  \struct XSurfY
   *  \brief Type traits for an y-surface field on a x-staggered volume
   *
   *  \struct XSurfZ
   *  \brief Type traits for an z-surface field on a x-staggered volume
   */
  struct XVol  { typedef NODIR FaceDir;  typedef IndexTriplet<-1, 0, 0> Offset;  typedef IndexTriplet<1,0,0> BCExtra; };
  struct XSurfX{ typedef XDIR  FaceDir;  typedef IndexTriplet< 0, 0, 0> Offset;  typedef IndexTriplet<0,0,0> BCExtra; };
  struct XSurfY{ typedef YDIR  FaceDir;  typedef IndexTriplet<-1,-1, 0> Offset;  typedef IndexTriplet<0,1,0> BCExtra; };
  struct XSurfZ{ typedef ZDIR  FaceDir;  typedef IndexTriplet<-1, 0,-1> Offset;  typedef IndexTriplet<0,0,1> BCExtra; };

  /**
   *  \struct YVol
   *  \brief Type traits for a y-staggered volume field
   *
   *  \struct YSurfX
   *  \brief Type traits for an x-surface field on a y-staggered volume
   *
   *  \struct YSurfY
   *  \brief Type traits for an y-surface field on a y-staggered volume
   *
   *  \struct YSurfZ
   *  \brief Type traits for an z-surface field on a y-staggered volume
   */
  struct YVol  { typedef NODIR FaceDir;  typedef IndexTriplet< 0,-1, 0> Offset;  typedef IndexTriplet<0,1,0> BCExtra; };
  struct YSurfX{ typedef XDIR  FaceDir;  typedef IndexTriplet<-1,-1, 0> Offset;  typedef IndexTriplet<1,0,0> BCExtra; };
  struct YSurfY{ typedef YDIR  FaceDir;  typedef IndexTriplet< 0, 0, 0> Offset;  typedef IndexTriplet<0,0,0> BCExtra; };
  struct YSurfZ{ typedef ZDIR  FaceDir;  typedef IndexTriplet< 0,-1,-1> Offset;  typedef IndexTriplet<0,0,1> BCExtra; };

  /**
   *  \struct ZVol
   *  \brief Type traits for a z-staggered volume field
   *
   *  \struct ZSurfX
   *  \brief Type traits for an x-surface field on a z-staggered volume
   *
   *  \struct ZSurfY
   *  \brief Type traits for an y-surface field on a z-staggered volume
   *
   *  \struct ZSurfZ
   *  \brief Type traits for an z-surface field on a z-staggered volume
   */
  struct ZVol  { typedef NODIR FaceDir;  typedef IndexTriplet< 0, 0,-1> Offset;  typedef IndexTriplet<0,0,1> BCExtra; };
  struct ZSurfX{ typedef XDIR  FaceDir;  typedef IndexTriplet<-1, 0,-1> Offset;  typedef IndexTriplet<1,0,0> BCExtra; };
  struct ZSurfY{ typedef YDIR  FaceDir;  typedef IndexTriplet< 0,-1,-1> Offset;  typedef IndexTriplet<0,1,0> BCExtra; };
  struct ZSurfZ{ typedef ZDIR  FaceDir;  typedef IndexTriplet< 0, 0, 0> Offset;  typedef IndexTriplet<0,0,0> BCExtra; };

  /**
   *  \struct SingleValue
   *  \brief Type traits for a single value field
   */
  struct SingleValue { typedef NODIR FaceDir; typedef IndexTriplet<0,0,0> Offset; typedef IndexTriplet<0,0,0> BCExtra; };

}// namespace SpatialOps

/**
 *  @}
 */

#endif
