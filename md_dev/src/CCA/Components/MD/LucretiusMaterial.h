/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#ifndef UINTAH_LUCRETIUS_MATERIAL_H
#define UINTAH_LUCRETIUS_MATERIAL_H

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Components/MD/MDMaterial.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

  class LucretiusMaterial : public MDMaterial {

    public:

      LucretiusMaterial(ProblemSpecP&, SimulationStateP& sharedState);

      virtual ~LucretiusMaterial();

      virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

      virtual void calculateForce();
      virtual void calculateEnergy();
      virtual double getCharge() const;
      virtual double getPolarizability() const;


    private:

      SCIRun::Vector d_force;
      double d_energy;
      double d_charge;
      double d_polarizability;

      // Prevent copying or assignment
      LucretiusMaterial(const LucretiusMaterial &material);
      LucretiusMaterial& operator=(const LucretiusMaterial &material);

  };

}  // End namespace Uintah

#endif // UINTAH_LUCRETIUS_MATERIAL_H
