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

#ifndef UINTAH_MD_MATERIAL_H
#define UINTAH_MD_MATERIAL_H

#include <CCA/Components/MD/Potentials/NonbondedPotential.h>
#include <CCA/Components/MD/MDFlags.h>

#include <Core/Grid/Material.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>


namespace Uintah {

  using namespace SCIRun;

  // Base class for MDMaterial, specific forcefield materials will derive from this
  class MDMaterial : public Material {

    public:

      MDMaterial();
//      MDMaterial(ProblemSpecP&, SimulationStateP& sharedState);
      virtual ~MDMaterial();

      virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

      virtual std::string getMaterialDescriptor() const = 0;

      // Provides the charge of the atom type if defined, 0 otherwise
      virtual double getCharge() const = 0;

      // Provides the polarizability of the atom type if defined, 0 otherwise
      virtual double getPolarizability() const = 0;

      // Provides the mass of the atom type (required)
      virtual double getMass() const = 0;

      // Provides the label the material uses to access potential maps
      virtual std::string getMapLabel() const = 0;

      // Provides the label the material uses to access this material itself
      virtual std::string getMaterialLabel() const = 0;

      // These labels differ as follows:
      //   getMapLabel will provide one part of a two-label nonbonded potential key pair.
      //       This map key is used to index into the non-bonded interactions.

      //   getMaterialLabel provides the label for the specific atom type being considered.

      // The difference is that getMaterialLabel will differentiate between atom types with the same exact
      //   nonbonded interaction:  e.g. in a forcefield with static charges for different atom types, two atom types
      //   may have different charges and/or polarizabilities, but still use the same nonbonded potential.  Uintah::MD
      //   will instantiate only one instance of this potential and share it with everything that uses it.  In this case,
      //   a material label must contain additional information to disambiguate the charge types; however, regardless of
      //   charge type, the map label will always point to the base nonbonded type and use that to determine the appropriate
      //   nonbonded potential.

      virtual NonbondedPotential* getPotentialHandle() const = 0;

    private:
      // Prevent copying or assignment
      MDMaterial(const MDMaterial &material);
      MDMaterial& operator=(const MDMaterial &material);

  };

}  // End namespace Uintah

#endif // UINTAH_MD_MATERIAL_H
