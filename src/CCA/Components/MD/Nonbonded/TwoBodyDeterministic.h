/*
 * TwoBodyDeterministic.h
 *
 *  Created on: Apr 6, 2014
 *      Author: jbhooper
 */

#ifndef TWOBODYDETERMINISTIC_H_
#define TWOBODYDETERMINISTIC_H_

#include <CCA/Components/MD/NonBonded.h>
#include <CCA/Components/MD/MDLabel.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {

  class TwoBodyDeterministic : public NonBonded {
    public:
     ~TwoBodyDeterministic() {}
      TwoBodyDeterministic(MDSystem*, MDLabel*, double);
      void initialize(const ProcessorGroup* pg,
                      const PatchSubset* perProcPatches,
                      const MaterialSubset* materials,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw);

      void setup(const ProcessorGroup* pg,
                 const PatchSubset* perProcPatches,
                 const MaterialSubset* materials,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw);

      void calculate(const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* materials,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     SchedulerP& subscheduler,
                     const LevelP& level);


      void finalize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* materials,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

      inline void setMDLabel(MDLabel* lb)
      {
        d_Label = lb;
      }

      inline std::string getNonbondedType() const {
        return nonbondedType;
      }
    private:
      static const std::string nonbondedType;
      MDSystem* d_System;
      MDLabel* d_Label;
      const double d_nonbondedRadius;

  };
}



#endif /* TWOBODYDETERMINISTIC_H_ */
