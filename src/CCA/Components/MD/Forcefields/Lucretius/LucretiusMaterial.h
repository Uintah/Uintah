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
 * Lucretius.h
 *
 *  Created on: Jan 28, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUS_H_
#define LUCRETIUS_MATERIAL_H

#include </CCA/Components/MD/MDMaterial.h>
#include </CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>

namespace Uintah {

  using namespace SCIRun;

  class LucretiusMaterial : public MDMaterial {
    public:
      LucretiusMaterial();
      LucretiusMaterial(ProblemSpecP&,
                        SimulationStateP& sharedState);
      ~LucretiusMaterial();

      ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

      inline void calculateForce(SCIRun::Vector& Force,
                                 const SCIRun::Vector& R) const {
        nonBondedPotential->fillForce(Force, R);
        return;
      }

      inline void calculateForce(SCIRun::Vector& Force,
                                 const SCIRun::Vector& R1,
                                 const SCIRun::Vector& R2) const {
        nonBondedPotential->fillForce(Force, R1, R2);
        return;
      }

      inline void calculateEnergy(double& Energy,
                                  const SCIRun::Vector& R) const {
        nonBondedPotential->fillEnergy(Energy, R);
        return;
      }

      inline void calculateEnergy(double& Energy,
                                  const SCIRun::Vector& R1,
                                  const SCIRun::Vector& R2) const {
        nonBondedPotential->fillEnergy(Energy, R1, R2);
        return;
      }

      inline void calculateForceAndEnergy(SCIRun::Vector& Force,
                                          double& Energy,
                                          const SCIRun::Vector& R) const {
        nonBondedPotential->fillEnergyAndForce(Force, Energy, R);
        return;
      }

      inline void calculateForceAndEnergy(SCIRun::Vector& Force,
                                          double& Energy,
                                          const SCIRun::Vector& R1,
                                          const SCIRun::Vector& R2) const {
        nonBondedPotential->fillEnergyAndForce(Force, Energy, R1, R2);
        return;
      }

      inline SCIRun::Vector getForce(const SCIRun::Vector& offSet) const {
        SCIRun::Vector Force;
        nonBondedPotential->fillForce(Force, offSet);
        return Force;
      }

      inline SCIRun::Vector getForce(const SCIRun::Vector& P1,
                                     const SCIRun::Vector& P2) const {
        SCIRun::Vector Force;
        nonBondedPotential->fillForce(Force, P1, P2);
        return Force;
      }

      inline double getEnergy(const SCIRun::Vector& offSet) const {
        double Energy;
        nonBondedPotential->fillEnergy(Energy, offSet);
        return Energy;
      }

      inline double getEnergy(const SCIRun::Vector& P1,
                              const SCIRun::Vector& P2) const {
        double Energy;
        nonBondedPotential->fillEnergy(Energy, P1, P2);
        return Energy;
      }

      inline double getCharge() const {
        return d_atomCharge;
      }

      inline double getPolarizability() const {
        return d_atomPolarizability;
      }

      inline std::string getNonbondedType() const {
        return nonBondedPotential->getPotentialDescriptor();
      }

    private:
      double d_atomCharge;
      double d_atomPolarizability;
      static const std::string materialClassDescriptor = "Lucretius";

      Uintah_MD::NonbondedTwoBodyPotential* nonBondedPotential;

      // Prevent copying or assignment
      LucretiusMaterial(const LucretiusMaterial& material);
      LucretiusMaterial& operator=(const LucretiusMaterial &material);
  };
}

#endif /* LUCRETIUS_MATERIAL_H_ */
