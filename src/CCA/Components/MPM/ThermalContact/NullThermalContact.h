/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __NullThermalContact__
#define __NullThermalContact__

#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <cmath>

namespace Uintah {
using namespace SCIRun;

   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;

/**************************************

CLASS
   NullThermalContact
   
   This version of thermal contact drives the temperatures
   of two materials to the same value at each grid point.

GENERAL INFORMATION

   NullThermalContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
 
KEYWORDS
   NullThermalContact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class NullThermalContact : public ThermalContact {
    public:
    // Constructor
    NullThermalContact(ProblemSpecP& ps,SimulationStateP& d_sS, MPMLabel* lb,
                       MPMFlags* MFlag);

    // Destructor
    virtual ~NullThermalContact();

    virtual void computeHeatExchange(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);
         
    virtual void initializeThermalContact(const Patch* patch,
                                int vfindex,
                                DataWarehouse* new_dw);

    virtual void addComputesAndRequires(Task* task,
                              const PatchSet* patches,
                              const MaterialSet* matls) const;

    virtual void outputProblemSpec(ProblemSpecP& ps);

    private:
      SimulationStateP d_sharedState;
      MPMLabel* lb;
      // Prevent copying of this class
      // copy constructor
      NullThermalContact(const NullThermalContact &con);
      NullThermalContact& operator=(const NullThermalContact &con);
  };
      
} // End namespace Uintah

#endif // __NullThermalContact__
