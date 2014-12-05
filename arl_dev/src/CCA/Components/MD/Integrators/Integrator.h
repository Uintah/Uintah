/*
 * Integrator.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <Core/Grid/Variables/ComputeSet.h>

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>

namespace Uintah {

  enum interactionModel { Deterministic, Stochastic, Mixed };
  class Integrator {

    public:
      Integrator();
      virtual ~Integrator();

      virtual void initialize(const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     atomTypes,
                                    DataWarehouse*    /*oldDW*/,
                                    DataWarehouse*      newDW,
                              const SimulationStateP*   simState,
                                    MDSystem*           systemInfo,
                              const MDLabel*            label,
                                    CoordinateSystem*   coordSys) = 0;

      virtual void setup(const ProcessorGroup*      pg,
                         const PatchSubset*         patches,
                         const MaterialSubset*      atomTypes,
                               DataWarehouse*       oldDW,
                               DataWarehouse*       newDW,
                         const SimulationStateP*    simState,
                               MDSystem*            systemInfo,
                         const MDLabel*             label,
                               CoordinateSystem*    coordSys) = 0;

      virtual void calculate(const ProcessorGroup*      pg,
                             const PatchSubset*         patches,
                             const MaterialSubset*      atomTypes,
                                   DataWarehouse*       oldDW,
                                   DataWarehouse*       newDW,
                             const SimulationStateP*    simState,
                                   MDSystem*            systemInfo,
                             const MDLabel*             label,
                                   CoordinateSystem*    coordSys) = 0;

      virtual void finalize(const ProcessorGroup*       pg,
                            const PatchSubset*          patches,
                            const MaterialSubset*       atomTypes,
                                  DataWarehouse*        oldDW,
                                  DataWarehouse*        newDW,
                            const SimulationStateP*     simState,
                                  MDSystem*             systemInfo,
                            const MDLabel*              label,
                                  CoordinateSystem*     coordSys) = 0;

//      virtual void advanceTime() const = 0;
      virtual std::string       getType()               const = 0;
      virtual interactionModel  getInteractionModel()   const = 0;
    private:
      // Forbid implicit instantiation and/or copy
      Integrator(const Integrator&);
      Integrator& operator=(const Integrator&);
//      MDIntegrator type;
  };
}



#endif /* INTEGRATOR_H_ */
