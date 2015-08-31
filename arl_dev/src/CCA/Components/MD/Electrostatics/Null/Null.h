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
 * Null.h
 *
 *  Created on: Sep 26, 2014
 *      Author: jbhooper
 */

#ifndef ELECTROSTATICNULL_H_
#define ELECTROSTATICNULL_H_

#include <CCA/Components/MD/MDSubcomponent.h>

#include <CCA/Components/MD/Electrostatics/Electrostatics.h>

namespace Uintah {

  class ElectrostaticNull: public Electrostatics, public MDSubcomponent
  {
    public:
      ElectrostaticNull();
     ~ElectrostaticNull();
     // Methods inherited from the Electrostatics interface

     virtual void initialize(const ProcessorGroup*      pg,
                             const PatchSubset*         patches,
                             const MaterialSubset*      materials,
                                   DataWarehouse*     /*oldDW*/,
                                   DataWarehouse*       newDW,
                             const SimulationStateP*  /*simState*/,
                                   MDSystem*          /*systemInfo*/,
                             const MDLabel*             label,
                                   CoordinateSystem*  /*coordinateSys*/);

     virtual void setup     (const ProcessorGroup*    /*pg*/,
                             const PatchSubset*       /*patches*/,
                             const MaterialSubset*    /*materials*/,
                                   DataWarehouse*     /*oldDW*/,
                                   DataWarehouse*     /*newDW*/,
                             const SimulationStateP*  /*simState*/,
                                   MDSystem*          /*systemInfo*/,
                             const MDLabel*           /*label*/,
                                   CoordinateSystem*  /*coordinateSys*/);

     virtual void calculate (const ProcessorGroup*      pg,
                             const PatchSubset*         patches,
                             const MaterialSubset*      materials,
                                   DataWarehouse*       oldDW,
                                   DataWarehouse*       newDW,
                             const SimulationStateP*  /*simState*/,
                                   MDSystem*          /*systemInfo*/,
                             const MDLabel*             label,
                                   CoordinateSystem*  /*coordinateSys*/,
                                   SchedulerP&        /*subscheduler*/,
                             const LevelP&              level);

     virtual void finalize  (const ProcessorGroup*    /*pg*/,
                             const PatchSubset*       /*patches*/,
                             const MaterialSubset*    /*materials*/,
                                   DataWarehouse*     /*oldDW*/,
                                   DataWarehouse*     /*newDW*/,
                             const SimulationStateP*  /*simState*/,
                                   MDSystem*          /*systemInfo*/,
                             const MDLabel*           /*label*/,
                                   CoordinateSystem*  /*coordinateSys*/);

     // Methods inherited from the MDSubcomponent interface
     virtual void registerRequiredParticleStates(
                                         varLabelArray&    particleState,
                                         varLabelArray&    particleState_preReloc,
                                         MDLabel*       label) const;

     virtual void addInitializeRequirements(Task*       task,
                                            MDLabel*    label) const;
     virtual void addInitializeComputes    (Task*       task,
                                            MDLabel*    label) const;
     virtual void addSetupRequirements     (Task*       task,
                                            MDLabel*    label) const;
     virtual void addSetupComputes         (Task*       task,
                                            MDLabel*    label) const;
     virtual void addCalculateRequirements (Task*       task,
                                            MDLabel*    label) const;
     virtual void addCalculateComputes     (Task*       task,
                                            MDLabel*    label) const;
     virtual void addFinalizeRequirements  (Task*       task,
                                            MDLabel*    label) const;
     virtual void addFinalizeComputes      (Task*       task,
                                            MDLabel*    label) const;

     virtual ElectrostaticsType getType() const
     {
       return Electrostatics::NONE;
     }

     virtual int requiredGhostCells() const
     {
       return 0;
     }

     virtual bool isPolarizable() const
     {
       return false;
     }

    private:
  };
}


#endif /* NULL_H_ */
