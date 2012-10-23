/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_HOMEBREW_RIGIDMPM_H
#define UINTAH_HOMEBREW_RIGIDMPM_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Components/MPM/Contact/Contact.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {

using namespace SCIRun;

class ThermalContact;

/**************************************

CLASS
   RigidMPM
   
   Short description...

GENERAL INFORMATION

   RigidMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   RigidMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class RigidMPM : public SerialMPM {
public:
  RigidMPM(const ProcessorGroup* myworld);
  virtual ~RigidMPM();

  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, 
                            GridP& grid, SimulationStateP&);
         
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;

  //////////
  // Insert Documentation Here:
  void computeStressTensor(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void scheduleComputeInternalForce(           SchedulerP&, const PatchSet*,
                                               const MaterialSet*);


  void computeInternalForce(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw);

  void scheduleComputeAndIntegrateAcceleration(SchedulerP&, const PatchSet*,
                                               const MaterialSet*);


  // Insert Documentation Here:
  virtual void computeAndIntegrateAcceleration(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);


  void scheduleInterpolateToParticlesAndUpdate(SchedulerP&,
                                                       const PatchSet*,
                                                       const MaterialSet*);

  void interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

private:
  RigidMPM(const RigidMPM&);
  RigidMPM& operator=(const RigidMPM&);
};
      
} // end namespace Uintah

#endif
