/*
 * TwoBodyDeterministic.h
 *
 *  Created on: Apr 6, 2014
 *      Author: jbhooper
 */

#ifndef TWOBODYDETERMINISTIC_H_
#define TWOBODYDETERMINISTIC_H_

#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/MDSubcomponent.h>
#include <CCA/Components/MD/MDUtil.h>

#include <CCA/Components/MD/Nonbonded/Nonbonded.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {

  class TwoBodyDeterministic : public Nonbonded, public MDSubcomponent {
    public:

// Constructors and destructors
     ~TwoBodyDeterministic() {}
      TwoBodyDeterministic(double, int);

// Inherited from nonbonded
      void initialize(  const ProcessorGroup*       pg,
                        const PatchSubset*          patches,
                        const MaterialSubset*       materials,
                        DataWarehouse*              oldDW,
                        DataWarehouse*              newDW,
                        SimulationStateP&           simState,
                        MDSystem*                   systemInfo,
                        const MDLabel*              label,
                        CoordinateSystem*           coordSys);

      void setup(       const ProcessorGroup*       pg,
                        const PatchSubset*          patches,
                        const MaterialSubset*       materials,
                        DataWarehouse*              oldDW,
                        DataWarehouse*              newDW,
                        SimulationStateP&           simState,
                        MDSystem*                   systemInfo,
                        const MDLabel*              label,
                        CoordinateSystem*           coordSys);

      void calculate(   const ProcessorGroup*       pg,
                        const PatchSubset*          patches,
                        const MaterialSubset*       materials,
                        DataWarehouse*              oldDW,
                        DataWarehouse*              newDW,
                        SimulationStateP&           simState,
                        MDSystem*                   systemInfo,
                        const MDLabel*              label,
                        CoordinateSystem*           coordSys);


      void finalize(    const ProcessorGroup*       pg,
                        const PatchSubset*          patches,
                        const MaterialSubset*       materials,
                        DataWarehouse*              oldDW,
                        DataWarehouse*              newDW,
                        SimulationStateP&           simState,
                        MDSystem*                   systemInfo,
                        const MDLabel*              label,
                        CoordinateSystem*           coordSys);

      inline std::string getNonbondedType() const {
        return nonbondedType;
      }

      inline int requiredGhostCells() const {
//        return SHRT_MAX;
        return d_nonbondedGhostCells;
      }

// Inherited from MDSubcomponent
      virtual void registerRequiredParticleStates(varLabelArray&,
                                                  varLabelArray&,
                                                  MDLabel*) const;

      virtual void addInitializeRequirements(Task*, MDLabel*) const;
      virtual void addInitializeComputes(Task*, MDLabel*) const;

      virtual void addSetupRequirements(Task*, MDLabel*) const;
      virtual void addSetupComputes(Task*, MDLabel*) const;

      virtual void addCalculateRequirements(Task*, MDLabel*) const;
      virtual void addCalculateComputes(Task*, MDLabel*) const;

      virtual void addFinalizeRequirements(Task*, MDLabel*) const;
      virtual void addFinalizeComputes(Task*, MDLabel*) const;


    private:
      static const std::string nonbondedType;
      const double d_nonbondedRadius;
      const int d_nonbondedGhostCells;

  };
}



#endif /* TWOBODYDETERMINISTIC_H_ */
