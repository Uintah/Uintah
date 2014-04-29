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

#ifndef LUCRETIUS_MATERIAL_H_
#define LUCRETIUS_MATERIAL_H



#include <CCA/Components/MD/MDMaterial.h>
#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>

namespace Uintah {

  using namespace SCIRun;

  class LucretiusMaterial : public MDMaterial {
    public:
      LucretiusMaterial();
//      LucretiusMaterial(ProblemSpecP&,
//                        SimulationStateP& sharedState);
      LucretiusMaterial(NonbondedTwoBodyPotential*, double, double, double, size_t);
      virtual ~LucretiusMaterial();

      virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

      virtual std::string getMaterialDescriptor() const {
        return ( materialClassDescriptor + "_" + nonbonded->getPotentialDescriptor() );
      }

      virtual double getCharge() const {
        return d_charge;
      }

      virtual double getPolarizability() const {
        return d_polarizability;
      }

      virtual double getMass() const {
        return d_mass;
      }

      virtual std::string getMapLabel() const {
        return nonbonded->getLabel();
      }

      virtual std::string getMaterialLabel() const {
        std::ostringstream outString;
        outString << nonbonded->getLabel() << d_subtypeNumber;
        return outString.str();
      }

      virtual NonbondedPotential* getPotentialHandle() const {
        return nonbonded;
      }


    private:
      double d_mass;
      double d_charge;
      double d_polarizability;
      size_t d_subtypeNumber;
      NonbondedTwoBodyPotential* nonbonded;

      static const std::string materialClassDescriptor;

      // Prevent copying or assignment
      LucretiusMaterial(const LucretiusMaterial& material);
      LucretiusMaterial& operator=(const LucretiusMaterial &material);
  };
}

#endif /* LUCRETIUS_MATERIAL_H_ */
