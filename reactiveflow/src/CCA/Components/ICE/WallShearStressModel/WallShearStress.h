/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef _WALLSHEARSTRESS_H
#define _WALLSHEARSTRESS_H

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationStateP.h>

#include <Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>


namespace Uintah {

  class DataWarehouse;
  class ICELabel;
  class Material;
  class Patch;
  

  class WallShearStress {

  public:
    WallShearStress();
    WallShearStress( ProblemSpecP& ps, SimulationStateP& sharedState);
    virtual ~WallShearStress(); 

    virtual void sched_Initialize(SchedulerP& sched, 
                                  const LevelP& level,
                                  const MaterialSet* matls) = 0;
    
    virtual void sched_AddComputeRequires(Task* task, 
                                          const MaterialSubset* matls) = 0;
                                               
    virtual
    void computeWallShearStresses( DataWarehouse* old_dw,
                                   DataWarehouse* new_dw,
                                   const Patch* patch,
                                   const int indx,
                                   constCCVariable<double>& vol_frac_CC,  
                                   constCCVariable<Vector>& vel_CC,      
                                   const CCVariable<double>& viscosity,        
                                   SFCXVariable<Vector>& tau_X_FC,
                                   SFCYVariable<Vector>& tau_Y_FC,
                                   SFCZVariable<Vector>& tau_Z_FC ) = 0;
  protected:
    
  };// End class WallShearStress

}// End namespace Uintah

#endif
