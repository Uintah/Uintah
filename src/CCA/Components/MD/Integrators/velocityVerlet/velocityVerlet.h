/*
 *
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
 *
 * ----------------------------------------------------------
 * velocityVerlet.h
 *
 *  Created on: Oct 17, 2014
 *      Author: jbhooper
 */

#ifndef VELOCITYVERLET_H_
#define VELOCITYVERLET_H_

#include <CCA/Components/MD/MDSubcomponent.h>

#include <CCA/Components/MD/Integrators/Integrator.h>

namespace Uintah{

  class velocityVerlet: public Integrator, public MDSubcomponent
  {
    public:
      velocityVerlet(const VarLabel* _dt_label);
     ~velocityVerlet();

// Method inherited from Integrator
      virtual void initialize(const ProcessorGroup*      pg,
                              const PatchSubset*         patches,
                              const MaterialSubset*      atomTypes,
                                    DataWarehouse*     /*oldDW*/,
                                    DataWarehouse*       newDW,
                              const SimulationStateP*    simState,
                                    MDSystem*            systemInfo,
                              const MDLabel*             label,
                                    CoordinateSystem*    coordSys);

      virtual void setup(     const ProcessorGroup*       pg,
                              const PatchSubset*          patches,
                              const MaterialSubset*       atomTypes,
                                    DataWarehouse*        oldDW,
                                    DataWarehouse*        newDW,
                              const SimulationStateP*     simState,
                                    MDSystem*             systemInfo,
                              const MDLabel*              label,
                                    CoordinateSystem*     coordSys);

      virtual void calculate( const ProcessorGroup*       pg,
                              const PatchSubset*          patches,
                              const MaterialSubset*       atomTypes,
                                    DataWarehouse*        oldDW,
                                    DataWarehouse*        newDW,
                              const SimulationStateP*     simState,
                                    MDSystem*             systemInfo,
                              const MDLabel*              label,
                                    CoordinateSystem*     coordSys);

      virtual void finalize(  const ProcessorGroup*       pg,
                              const PatchSubset*          patches,
                              const MaterialSubset*       atomTypes,
                                    DataWarehouse*        oldDW,
                                    DataWarehouse*        newDW,
                              const SimulationStateP*     simState,
                                    MDSystem*             systemInfo,
                              const MDLabel*              label,
                                    CoordinateSystem*     coordSys);

// Methods inherited from MDSubcomponent
      virtual void registerRequiredParticleStates(
                                         LabelArray& particleState,
                                         LabelArray& particleState_preReloc,
                                         MDLabel*    labels) const;

      virtual void addInitializeRequirements(Task*       task,
                                            MDLabel*    labels) const;
      virtual void addInitializeComputes(    Task*       task,
                                            MDLabel*    labels) const;

      virtual void addSetupRequirements(     Task*       task,
                                            MDLabel*    labels) const;
      virtual void addSetupComputes(         Task*       task,
                                            MDLabel*    labels) const;

      virtual void addCalculateRequirements( Task*       task,
                                            MDLabel*    labels) const;
      virtual void addCalculateComputes(     Task*       task,
                                            MDLabel*    labels) const;

      virtual void addFinalizeRequirements(  Task*       task,
                                            MDLabel*    labels) const;
      virtual void addFinalizeComputes(      Task*       task,
                                            MDLabel*    labels) const;

      virtual std::string getType() const
      {
        return d_integratorType;
      }

      virtual interactionModel getInteractionModel() const
      {
        return Deterministic;
      }
// Local methods
      inline double getKineticEnergy() {
        return d_previousKE;
      }

      inline double getPotentialEnergy() {
        return d_previousPE;
      }

      inline double getTotalEnergy() {
        return d_previousKE + d_previousPE;
      }

      inline SCIRun::Vector getBoxMomentum() {
        return d_previousMomentum;
      }

    private:
      void firstIntegrate(const PatchSubset*        patches,
                          const MaterialSubset*     materials,
                                DataWarehouse*      oldDW,
                                DataWarehouse*      newDW,
                          const SimulationStateP*   simState,
                          const MDLabel*            label);

      void integrate(     const PatchSubset*        patches,
                          const MaterialSubset*     materials,
                                DataWarehouse*      oldDW,
                                DataWarehouse*      newDW,
                          const SimulationStateP*   simState,
                          const MDLabel*            label);

      void firstIntegratePatch(const Patch*             patch,
                               const int                atomType,
                                     DataWarehouse*     oldDW,
                                     DataWarehouse*     newDW,
                               const SimulationStateP*  simState,
                               const MDLabel*           label,
                                     double&            kineticEnergy,
                                     double&            totalMass,
                                     SCIRun::Vector&    totalMomentum);

      void integratePatch(const Patch*             patch,
                          const int                atomType,
                                DataWarehouse*     oldDW,
                                DataWarehouse*     newDW,
                          const SimulationStateP*  simState,
                          const MDLabel*           label,
                                double&            kineticEnergy,
                                double&            totalMass,
                                SCIRun::Vector&    totalMomentum);

      const VarLabel*   dt_label;
      const std::string d_integratorType;

      bool d_firstIntegration;
      double d_previousKE;
      double d_previousPE;
      double d_previousTemp;
      double d_previousMass;
      SCIRun::Vector d_previousMomentum;
      double d_dt;
      int    d_currentTimestep;

  };
}


#endif /* VELOCITYVERLET_H_ */
