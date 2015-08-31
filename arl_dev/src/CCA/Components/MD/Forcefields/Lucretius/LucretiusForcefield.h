/*
 * LucretiusForcefield.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUSFORCEFIELD_H_
#define LUCRETIUSFORCEFIELD_H_

#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>

//#include <Core/Malloc/Allocator.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/MDMaterial.h>
#include <CCA/Components/MD/MDUtil.h>
#include <CCA/Components/MD/MDUnits.h>

#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>

#include <CCA/Components/MD/Forcefields/TwoBodyForceField.h>
#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>
#include <CCA/Components/MD/Forcefields/nonbondedPotentialMapKey.h>

#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusForcefield.h>
#include <CCA/Components/MD/Forcefields/Lucretius/nonbondedLucretius.h>
#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusMaterial.h>

#include <vector>
/*
 * ....................................................................................................................*
 */

namespace Uintah {

  class LucretiusForcefield : public TwoBodyForcefield {
    public:
//      LucretiusForcefield() {};
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

     virtual void registerAtomTypes(const varLabelArray&   particleState,
                                    const varLabelArray&   particleState_preReloc,
                                    const MDLabel*      label,
                                    SimulationStateP&   simState) const;

     virtual double ffDistanceToInternal() const
     {
       // Internal:  1 A = 1e-10 m
       // Lucretius: 1 A = 1e-10 m
       return 1.0;
     }

     virtual double ffTimeToInternal() const
     {
       // Internal:  1fs = 1e-15  s
       // Lucretius: 1fs = 1e-15 s
       return 1.0;
     }

     virtual double ffVelocityToInternal() const
     {
       // Internal:   A / fs
       // Lucretius:  m /  s

       return 1.0;//1e-5;
     }

     virtual double ffAccelerationToInternal() const
     {
       // Internal:  A / fs^2
       // Lucretius: A / fs^2
       // Note:  Only extended lagrangian quantities ever have explicit accelerations
       //        in a Lucretius forcefield implementation.
       return 1.0;
     }

     virtual double ffChargeToInternal() const
      {
        // Internal:  1e
        // Lucretius: 1e
        return 1.0;
      }

     virtual double ffEnergyToInternal() const
     {
       // Internal:  J/mol
       // Lucretius: kCal/mol
       // Lucretius->Internal = 4184.0
       return 1.0;  //4184.0;  //Convert from kCal/mol to J/mol

     }

     virtual double ffStressToInternal() const
     {
       // Internal   1 g / nm * ns ^ 2
       // Lucretius  atm
       // Lucretius->Internal = L/MDUnits::pressureToAtm();

       return 1.0 / MDUnits::pressureToAtm();
     }

     virtual double ffMassToInternal() const
     {
       // Internal  1 g
       // Lucretius 1 g/mol
       // Lucretius->Internal = L/MDUnits::molesPerAtom
       return 1.0;
     }

    private:
      // Private functions related to parsing of the input forcefield file
      NonbondedTwoBodyPotential* parseHomoatomicNonbonded(std::string&,
                                                          const forcefieldType,
                                                          double&);
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
