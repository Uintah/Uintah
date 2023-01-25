/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_HOMEBREW_ModelInterface_H
#define UINTAH_HOMEBREW_ModelInterface_H

#include <Core/Parallel/UintahParallelComponent.h>

#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

/**************************************

CLASS
   ModelInterface
   
   Short description...

GENERAL INFORMATION

   ModelInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Model of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Model_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

namespace Uintah {

  class ApplicationInterface;
  class Regridder;
  class Output;

  class DataWarehouse;
  class Material;
  class ProcessorGroup;
  class VarLabel;
  
  //________________________________________________
  class ModelInterface : public UintahParallelComponent {
  public:
    ModelInterface(const ProcessorGroup* myworld,
                   const MaterialManagerP materialManager);

    virtual ~ModelInterface();

    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp ) {};
    virtual void setComponents( ApplicationInterface *comp );
    virtual void getComponents();
    virtual void releaseComponents();
      
    virtual void problemSetup(GridP& grid, const bool isRestart) = 0;
      
    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

    virtual void scheduleInitialize(SchedulerP& scheduler,
                                    const LevelP& level) = 0;

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimeStep(SchedulerP& scheduler,
                                               const LevelP& level) = 0;
    
    virtual void scheduleRefine(const PatchSet* patches,
                                SchedulerP& sched) {};

    virtual void setAMR(bool val) { m_AMR = val; }
    virtual bool isAMR() const { return m_AMR; }
  
    virtual void setDynamicRegridding(bool val) {m_dynamicRegridding = val; }
    virtual bool isDynamicRegridding() const { return m_dynamicRegridding; }

  protected:
    ApplicationInterface   * m_application {nullptr};
    Scheduler              * m_scheduler   {nullptr};
    Regridder              * m_regridder   {nullptr};
    Output                 * m_output      {nullptr};
   
    MaterialManagerP m_materialManager {nullptr};
    
    bool m_AMR {false};
    bool m_dynamicRegridding {false};

  private:     
    ModelInterface(const ModelInterface&);
    ModelInterface& operator=(const ModelInterface&);
  };
} // End namespace Uintah
   
#endif
