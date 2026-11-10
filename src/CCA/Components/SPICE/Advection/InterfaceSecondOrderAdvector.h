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

#ifndef UINTAH_SPICE_INTERFACE_SECOND_ORDER_ADVECTOR_H
#define UINTAH_SPICE_INTERFACE_SECOND_ORDER_ADVECTOR_H

#include <CCA/Components/ICE/Advection/SecondOrderAdvector.h>

namespace Uintah {

/*______________________________________________________________________
 InterfaceSecondOrderAdvector: ICE's SecondOrderAdvector plus a per-face
 override of the slab value used by the vol_frac advectQ (through ICE's
 SecondOrderAdvector::adjustSlabs hook).  FSPICE + <interfaceTransport>
 donorAcceptor: the MPM materials' vol_frac in the pressure equation
 (SPICE::computeDelPressAndUpdatePressCC, SPICE::d_mpmAdvector) carries the
 complementary part of the gas share of each gas/solid face, so the face
 partition sums to 1.
______________________________________________________________________*/

  class InterfaceSecondOrderAdvector : public SecondOrderAdvector {

  public:
    InterfaceSecondOrderAdvector();
    InterfaceSecondOrderAdvector( DataWarehouse * new_dw,
                                  const Patch   * patch,
                                  const bool      isNewGrid );

    virtual ~InterfaceSecondOrderAdvector();

    virtual InterfaceSecondOrderAdvector* clone( DataWarehouse * new_dw,
                                                 const Patch   * patch,
                                                 const bool      isNewGrid ) override;

    //  Face value >= 0 replaces the slab value on both sides of that face;
    //  < 0 leaves it alone.  nullptr (the default) = stock SecondOrder.
    //  Arrays must cover the patch's faces plus one ghost layer.
    void setFaceFractions( const SFCXVariable<double> * faceFracX,
                           const SFCYVariable<double> * faceFracY,
                           const SFCZVariable<double> * faceFracZ )
    {
      d_faceFracX = faceFracX;
      d_faceFracY = faceFracY;
      d_faceFracZ = faceFracZ;
    };

  protected:
    virtual void adjustSlabs( CCVariable<facedata<double> > & q_OAFS,
                              const Patch * patch ) override;

  private:
    const SFCXVariable<double> * d_faceFracX{nullptr};
    const SFCYVariable<double> * d_faceFracY{nullptr};
    const SFCZVariable<double> * d_faceFracZ{nullptr};
  };
} // end namespace Uintah

#endif
