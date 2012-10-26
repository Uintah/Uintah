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

#ifndef __ThermalContact__
#define __ThermalContact__

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
  class MPMFlags;
  class MPMLabel;
  class ProcessorGroup;
  class Patch;
  class Task;
  class VarLabel;

/**************************************

CLASS
   ThermalContact
   
   Short description...

GENERAL INFORMATION

   ThermalContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   ThermalContact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class ThermalContact {
  public:
    // Constructor
    ThermalContact();
    virtual ~ThermalContact();

    virtual void computeHeatExchange(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw) = 0;
         
    virtual void initializeThermalContact(const Patch* patch,
                                int vfindex,
                                DataWarehouse* new_dw) = 0;

    virtual void addComputesAndRequires(Task* task,
                              const PatchSet* patches,
                              const MaterialSet* matls) const = 0;

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

  protected:
    MPMFlags* flag;
  private:
    MPMLabel* lb;
    
  };
      
} // End namespace Uintah

#endif // __ThermalContact__
