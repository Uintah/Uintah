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

#include "Grid.h"
#include <spatialops/structured/FVStaggeredFieldTypes.h>

namespace SpatialOps{

  template< typename DirT, typename FieldT >
  double shift(){
    // obtain the coordinate shift for the given direction.
    // note that (0,0,0) shifting corresponds to a (dx/2,dy/2,dz/2)
    // offset whereas (-1,-1,-1) shifting corresponds to (0,0,0).
    return 0.5*double(1+IndexStagger<FieldT,DirT>::value);
  };

  Grid::Grid( const IntVec npts,
              const DoubleVec length )
    : npts_( npts ),
      length_( length ),
      spacing_( length_/npts_ )
  {}

  template<typename CoordT> unsigned int get_dir();
  template<> unsigned int get_dir<XDIR>(){ return 0; }
  template<> unsigned int get_dir<YDIR>(){ return 1; }
  template<> unsigned int get_dir<ZDIR>(){ return 2; }

  //------------------------------------------------------------------

  template< typename CoordT >
  double Grid::spacing() const
  {
    return spacing_[ get_dir<CoordT>() ];
  }

  //------------------------------------------------------------------

  template< typename CoordT, typename FieldT >
  void Grid::set_coord( FieldT& f ) const
  {
    const unsigned int dir = get_dir<CoordT>();
    const double offset = shift<CoordT,FieldT>() * spacing_[dir];

    typedef typename FieldT::iterator FieldIter;

#   ifdef ENABLE_CUDA
    // if the field is on GPU, move it to CPU, populate it, then sync it back.
    // this is slow, but the Grid class isn't used much in production, and this
    // could be done only during the setup phase rather than repeatedly.
    const short devIx = f.active_device_index();
    const bool isCPU = (devIx == CPU_INDEX);
    if( !isCPU ) {
      f.add_device( CPU_INDEX );
      f.set_device_as_active( CPU_INDEX );
    }
#   endif

    FieldIter iter=f.begin();

    const MemoryWindow& mwInterior = f.window_without_ghost();
    const MemoryWindow& mw         = f.window_with_ghost();

    const IntVec lo(0,0,0);
    const IntVec hi( mw.extent() );

    const int ixOffset = mwInterior.offset(dir);
    IntVec ix;
    for( ix[2]=lo[2]; ix[2]<hi[2]; ++ix[2] ){
      for( ix[1]=lo[1]; ix[1]<hi[1]; ++ix[1] ){
        for( ix[0]=lo[0]; ix[0]<hi[0]; ++ix[0] ){
          *iter = spacing_[dir] * (ix[dir]-ixOffset) + offset;
          ++iter;
        }
      }
    }
#   ifdef ENABLE_CUDA
    if( !isCPU ){
      f.validate_device( devIx );
      f.set_device_as_active( devIx );
    }
#   endif
  }

  //==================================================================
  // Explicit template instantiation
  //
# define DECLARE_COORD( FIELD )                                 \
  template void Grid::set_coord< XDIR, FIELD >( FIELD& ) const;	\
  template void Grid::set_coord< YDIR, FIELD >( FIELD& ) const;	\
  template void Grid::set_coord< ZDIR, FIELD >( FIELD& ) const;

# define DECLARE( VOL )                                 \
  DECLARE_COORD( VOL );                                 \
  DECLARE_COORD( FaceTypes<VOL>::XFace );               \
  DECLARE_COORD( FaceTypes<VOL>::YFace );               \
  DECLARE_COORD( FaceTypes<VOL>::ZFace );

  DECLARE( SVolField );
  DECLARE( XVolField );
  DECLARE( YVolField );
  DECLARE( ZVolField );

  template double Grid::spacing<XDIR>() const;
  template double Grid::spacing<YDIR>() const;
  template double Grid::spacing<ZDIR>() const;
  //
  //==================================================================

} // namespace SpatialOps
