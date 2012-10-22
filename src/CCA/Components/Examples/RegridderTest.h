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

#ifndef Packages_Uintah_CCA_Components_Examples_RegridderTest_h
#define Packages_Uintah_CCA_Components_Examples_RegridderTest_h

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Geometry/Vector.h>

namespace Uintah
{
  class SimpleMaterial;
  class ExamplesLabel;
  class VarLabel;
/**************************************

CLASS
   RegridderTest
   
   RegridderTest simulation

GENERAL INFORMATION

   RegridderTest.h

   Randy N. Jones
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Regridder

DESCRIPTION
   This is a simple component that will test Oren Livne's regridding
   algorith when using AMR.
  
WARNING
  
****************************************/

  class RegridderTest: public UintahParallelComponent, public SimulationInterface {
  public:
    RegridderTest ( const ProcessorGroup* myworld );
    virtual ~RegridderTest ( void );

    // Interface inherited from Simulation Interface
    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP& state );
    virtual void scheduleInitialize            ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleTimeAdvance           ( const LevelP& level, SchedulerP& scheduler);
    virtual void scheduleErrorEstimate         ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleInitialErrorEstimate  ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleCoarsen               ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleRefine                ( const PatchSet* patches, SchedulerP& scheduler );
    virtual void scheduleRefineInterface       ( const LevelP& level, SchedulerP& scheduler, bool needCoarseOld, bool needCoarseNew);

  private:
    void initialize ( const ProcessorGroup*,
		      const PatchSubset* patches, const MaterialSubset* matls,
		      DataWarehouse* old_dw, DataWarehouse* new_dw );

    void computeStableTimestep ( const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw, DataWarehouse* new_dw );

    void timeAdvance ( const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw);

    void errorEstimate ( const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse*, DataWarehouse* new_dw, bool initial);

    void coarsen ( const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse*, DataWarehouse* new_dw);

    void refine ( const ProcessorGroup*,
		  const PatchSubset* patches,
		  const MaterialSubset* matls,
		  DataWarehouse*, DataWarehouse* new_dw);

    ExamplesLabel*   d_examplesLabel;
    SimulationStateP d_sharedState;
    SimpleMaterial*  d_material;

    VarLabel* d_oldDensityLabel;
    VarLabel* d_densityLabel;
    VarLabel* d_currentAngleLabel;
    SCIRun::Vector d_gridMax;
    SCIRun::Vector d_gridMin;

    // Fake cylinder
    SCIRun::Vector d_centerOfBall;
    SCIRun::Vector d_centerOfDomain;
    SCIRun::Vector d_oldCenterOfBall;
    double         d_radiusOfBall;
    double         d_radiusOfOrbit;
    double         d_angularVelocity;

    bool           d_radiusGrowth;
    bool           d_radiusGrowthDir;
  };

} // namespace Uintah

#endif
