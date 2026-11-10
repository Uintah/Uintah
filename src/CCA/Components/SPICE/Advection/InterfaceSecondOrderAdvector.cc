/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/SPICE/Advection/InterfaceSecondOrderAdvector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

InterfaceSecondOrderAdvector::InterfaceSecondOrderAdvector()
{
}

InterfaceSecondOrderAdvector::InterfaceSecondOrderAdvector( DataWarehouse * new_dw,
                                                            const Patch   * patch,
                                                            const bool      isNewGrid )
  : SecondOrderAdvector( new_dw, patch, isNewGrid )
{
}

InterfaceSecondOrderAdvector::~InterfaceSecondOrderAdvector()
{
}

InterfaceSecondOrderAdvector*
InterfaceSecondOrderAdvector::clone( DataWarehouse * new_dw,
                                     const Patch   * patch,
                                     const bool      isNewGrid )
{
  return scinew InterfaceSecondOrderAdvector( new_dw, patch, isNewGrid );
}

//______________________________________________________________________
//  FSPICE no-penetration face partition (SPICE::computeDelPressAndUpdatePressCC):
//  a face value >= 0 replaces the slab value on both sides of that face;
//  only the donor's slab carries volume, so that is the value transported.
void
InterfaceSecondOrderAdvector::adjustSlabs( CCVariable<facedata<double> > & q_OAFS,
                                           const Patch * patch )
{
  if( d_faceFracX ){
    const SFCXVariable<double>& oX = *d_faceFracX;
    const SFCYVariable<double>& oY = *d_faceFracY;
    const SFCZVariable<double>& oZ = *d_faceFracZ;

    auto inWindow = []( const IntVector & f, const IntVector & l, const IntVector & h ){
      return f.x() >= l.x() && f.y() >= l.y() && f.z() >= l.z() &&
             f.x() <  h.x() && f.y() <  h.y() && f.z() <  h.z();
    };
    auto setSlab = [&]( facedata<double> & slab, int face, double ovr ){
      if( ovr >= 0.0 ){
        slab.fdata[face] = ovr;
      }
    };

    for(CellIterator iter = patch->getExtraCellIterator(1); !iter.done(); iter++){
      const IntVector& c = *iter;
      facedata<double>& slab = q_OAFS[c];
      const IntVector xp = c + IntVector(1,0,0);
      const IntVector yp = c + IntVector(0,1,0);
      const IntVector zp = c + IntVector(0,0,1);

      if( inWindow( c,  oX.getLowIndex(), oX.getHighIndex() ) ){ setSlab( slab, LEFT,   oX[c]  ); }
      if( inWindow( xp, oX.getLowIndex(), oX.getHighIndex() ) ){ setSlab( slab, RIGHT,  oX[xp] ); }
      if( inWindow( c,  oY.getLowIndex(), oY.getHighIndex() ) ){ setSlab( slab, BOTTOM, oY[c]  ); }
      if( inWindow( yp, oY.getLowIndex(), oY.getHighIndex() ) ){ setSlab( slab, TOP,    oY[yp] ); }
      if( inWindow( c,  oZ.getLowIndex(), oZ.getHighIndex() ) ){ setSlab( slab, BACK,   oZ[c]  ); }
      if( inWindow( zp, oZ.getLowIndex(), oZ.getHighIndex() ) ){ setSlab( slab, FRONT,  oZ[zp] ); }
    }
  }
}
