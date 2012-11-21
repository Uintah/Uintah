/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_MDLABEL_H
#define UINTAH_MDLABEL_H

namespace Uintah {

class VarLabel;

class MDLabel {
  public:

    MDLabel();
    ~MDLabel();

    // vector quantities
    const VarLabel* pXLabel;
    const VarLabel* pXLabel_preReloc;

    const VarLabel* pForceLabel;
    const VarLabel* pForceLabel_preReloc;

    const VarLabel* pAccelLabel;
    const VarLabel* pAccelLabel_preReloc;

    const VarLabel* pVelocityLabel;
    const VarLabel* pVelocityLabel_preReloc;

    // scalars
    const VarLabel* pEnergyLabel;
    const VarLabel* pEnergyLabel_preReloc;

    const VarLabel* pMassLabel;
    const VarLabel* pMassLabel_preReloc;

    const VarLabel* pChargeLabel;
    const VarLabel* pChargeLabel_preReloc;

    const VarLabel* pParticleIDLabel;
    const VarLabel* pParticleIDLabel_preReloc;

    // reduction variables
    const VarLabel* vdwEnergyLabel;

};
}  // End namespace Uintah

#endif
