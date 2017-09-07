/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#include <CCA/Components/Wasatch/Operators/Extrapolant.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace WasatchCore{

//--------------------------------------------------------------------

template< typename FieldT >
Extrapolant<FieldT>::
Extrapolant( const std::vector<bool>& bcMinus,
             const std::vector<bool>& bcPlus )
: bcMinus_ (bcMinus),
  bcPlus_  (bcPlus )
{
  unitNormal_.push_back(SpatialOps::IntVec(1,0,0));
  unitNormal_.push_back(SpatialOps::IntVec(0,1,0));
  unitNormal_.push_back(SpatialOps::IntVec(0,0,1));
}

//--------------------------------------------------------------------

template< typename FieldT >
Extrapolant<FieldT>::
~Extrapolant()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Extrapolant<FieldT>::
apply_to_field( FieldT& src,
                const double minVal,
                const double maxVal,
                const bool skipBCs )
{
  // extrapolate from interior cells:
  using namespace SpatialOps;

  bool doMinMaxCheck = (minVal > DBLMIN || maxVal < DBLMAX);
  const MemoryWindow& ws = src.window_with_ghost();
    
  int pm[2]={1,-1}; // plus or minus face
  int zo[2]={0,1};  // zero and one
  bool skipBCFace;
  for (int face=0; face<3; face++) { // x, y, z faces
    for (int direc=0; direc<2; direc++) { // minus, plus direction
      skipBCFace = (direc == 0) ? bcMinus_[face] : bcPlus_[face]; // is the current face a physical boundary?
      if (skipBCFace && skipBCs) continue;
      const IntVec extent = ws.extent() - unitNormal_[face]*ws.glob_dim() + unitNormal_[face];
      const IntVec baseOffset = ws.offset() + (unitNormal_[face]*ws.glob_dim() - unitNormal_[face] )* zo[direc];
      
      const MemoryWindow wd( ws.glob_dim(), baseOffset, extent );
      
      const MemoryWindow ws1( ws.glob_dim(),
                             baseOffset + unitNormal_[face] * pm[direc],
                             extent );
      
      const MemoryWindow ws2( ws.glob_dim(),
                             baseOffset  + unitNormal_[face] * pm[direc] * 2,
                             extent );
      
      FieldT d( wd, src );
      const FieldT s1( ws1, src );
      const FieldT s2( ws2, src );

      if( doMinMaxCheck ){
        d <<= max( min( 2.0*s1 - s2, maxVal), minVal);
      }
      else{
        d <<= 2.0 * s1 - s2;
      }
      
    }
  }
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class Extrapolant< SpatialOps::SVolField >;
template class Extrapolant< SpatialOps::XVolField >;
template class Extrapolant< SpatialOps::YVolField >;
template class Extrapolant< SpatialOps::ZVolField >;
//==================================================================

}
