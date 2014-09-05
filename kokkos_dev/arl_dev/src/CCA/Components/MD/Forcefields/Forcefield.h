/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

/*
 * Forcefield.h
 *
 *  Created on: Feb 20, 2014
 *      Author: jbhooper
 */

#ifndef UINTAH_MD_FORCEFIELD_H_
#define UINTAH_MD_FORCEFIELD_H_

#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/MDUtil.h>

#include <CCA/Components/MD/Potentials/NonbondedPotential.h>
#include <CCA/Components/MD/Potentials/Valence/BondPotential.h>
#include <CCA/Components/MD/Potentials/Valence/BendPotential.h>
#include <CCA/Components/MD/Potentials/Valence/DihedralPotential.h>
#include <CCA/Components/MD/Potentials/Valence/ImproperDihedral.h>

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <vector>

namespace Uintah {
  enum forcefieldInteractionClass { TwoBody, ThreeBody, NBody };

  class Forcefield {
    public:
      Forcefield () {}
      virtual ~Forcefield () {}
      virtual forcefieldInteractionClass getInteractionClass() const = 0;
      virtual std::string getForcefieldDescriptor() const = 0;
      virtual void registerAtomTypes(const LabelArray&  particleState,
                                     const LabelArray&  particleState_preReloc,
                                     const MDLabel*     label,
                                     SimulationStateP&  simState) const = 0;

    private:

  };
}



#endif /* UINTAH_MD_FORCEFIELD_H_ */
