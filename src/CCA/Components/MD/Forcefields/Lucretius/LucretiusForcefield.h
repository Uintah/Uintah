/*
 * LucretiusForcefield.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUSFORCEFIELD_H_
#define LUCRETIUSFORCEFIELD_H_

#include <Core/Grid/SimulationState.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/MDUtil.h>


#include <CCA/Components/MD/Forcefields/TwoBodyForceField.h>
#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>
#include <CCA/Components/MD/Forcefields/nonbondedPotentialMapKey.h>

#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>
#include <CCA/Components/MD/Forcefields/Lucretius/nonbondedLucretius.h>
#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusMaterial.h>

#include <vector>
/*
 * ....................................................................................................................*
 */

namespace Uintah {

  class LucretiusForcefield : public TwoBodyForcefield {
    public:
      LucretiusForcefield() {};
      LucretiusForcefield(const ProblemSpecP& ps, SimulationStateP& sharedState);
     ~LucretiusForcefield() {};
     inline BondPotential* getBondPotential(int Index) const {
       return bonds[Index];
     }
     inline BendPotential* getBendPotential(int Index) const {
       return bends[Index];
     }
     inline DihedralPotential* getDihedralPotential(int Index) const {
       return dihedrals[Index];
     }

     inline ImproperDihedralPotential* getImproperDihedralPotential(int Index) const {
       return improper[Index];
     }

     NonbondedTwoBodyPotential* getNonbondedPotential(const std::string& label1, const std::string& label2) const;

     inline std::string getForcefieldDescriptor() const {
       return d_forcefieldNameString;
      }

     virtual void registerProvidedParticleStates(std::vector<const VarLabel*>&,
                                                   std::vector<const VarLabel*>&,
                                                   MDLabel*) const;

    private:
      // Private functions related to parsing of the input forcefield file
      NonbondedTwoBodyPotential* parseHomoatomicNonbonded(std::string&,
                                                          const forcefieldType,
                                                          double);
      NonbondedTwoBodyPotential* parseHeteroatomicNonbonded(std::string&,
                                                            const forcefieldType);
      void parseNonbondedPotentials(std::ifstream&,
                                    const std::string&,
                                    std::string&,
                                    SimulationStateP&);

      // Data members
      static const std::string d_forcefieldNameString;
      bool hasPolarizability;
      double tholeParameter;
      std::vector<LucretiusMaterial*> materialArray;
      nonbondedTwoBodyMapType potentialMap;
      std::vector<BondPotential*> bonds;
      std::vector<BendPotential*> bends;
      std::vector<DihedralPotential*> dihedrals;
      std::vector<ImproperDihedralPotential*> improper;

  };
}



#endif /* LUCRETIUSFORCEFIELD_H_ */
