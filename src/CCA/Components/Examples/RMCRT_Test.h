/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_Examples_RMCRT_Test_h
#define Packages_Uintah_CCA_Components_Examples_RMCRT_Test_h

#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/DebugStream.h>
#include <Core/GeometryPiece/GeometryPiece.h>

using Uintah::DebugStream;

namespace Uintah
{
  class SimpleMaterial;
  class ExamplesLabel;
  class VarLabel;
  class GeometryObject;
  class Ray;
  class floatRay;
/**************************************

CLASS
   RMCRT_Test
   
   RMCRT_Test simulation

GENERAL INFORMATION

   RMCRT_Test.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

  
   
KEYWORDS
   Regridder

DESCRIPTION
   Component for testing RMCRT concepts on multiple levels
  
WARNING
  
****************************************/

  class RMCRT_Test: public UintahParallelComponent, public SimulationInterface {

  public:

    RMCRT_Test ( const ProcessorGroup* myworld );
    virtual ~RMCRT_Test ( void );

    // Interface inherited from Simulation Interface
    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP& state );
    virtual void scheduleInitialize            ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleRestartInitialize     ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleTimeAdvance           ( const LevelP& level, SchedulerP& scheduler);
    virtual void scheduleInitialErrorEstimate  ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleCoarsen               ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleRefine                ( const PatchSet* patches, SchedulerP& scheduler );
    virtual void scheduleRefineInterface       ( const LevelP& level, SchedulerP& scheduler, bool needCoarseOld, bool needCoarseNew);

    virtual int computeTaskGraphIndex();
  

  private:
    void initialize ( const ProcessorGroup*,
                      const PatchSubset* patches, 
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw );
                      
    void initializeWithUda (const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* ,
                            DataWarehouse*,
                            DataWarehouse* new_dw);

    void computeStableTimestep ( const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw );

    //__________________________________
    //                                                
    void schedulePseudoCFD(SchedulerP& sched,
                           const PatchSet* patches,
                           const MaterialSet* matls);

    void pseudoCFD ( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);
                     
    //__________________________________
    // initialize the temperature/abskg/abskp 
    void sched_initProperties( const LevelP&, 
                               SchedulerP& sched );

    void initProperties( const ProcessorGroup* pc, 
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw );
    //__________________________________
    //                     
    void areGridsEqual( const GridP& uda_grid, 
                        const GridP& grid );

   protected:
    
    Ray* d_RMCRT{nullptr};
    
    SimulationStateP d_sharedState;
    SimpleMaterial*  d_material{nullptr};

    VarLabel* d_colorLabel{nullptr};
    VarLabel* d_divQLabel{nullptr};
    VarLabel* d_compAbskgLabel{nullptr};
    VarLabel* d_cellTypeLabel{nullptr};
    
    Ghost::GhostType d_gn;
    Ghost::GhostType d_gac;
    
    double   d_initColor{-9.0};
    double   d_initAbskg{-9.0};
    
    int      d_radCalc_freq{1};
    int      d_matl;
    int      d_wall_cell; 
    int      d_flow_cell;
    int      d_whichAlgo;
    int      d_benchmark{0};
    int      d_pDir;             // principle direction used for 1D benchmark tests

    enum Algorithm{ dataOnion,            
                    coarseLevel, 
                    singleLevel, 
                    radiometerOnly       // VRFlux is computed at radiometer locations
                  }; 
    
    std::vector<GeometryPieceP>  d_intrusion_geom;
    
    struct useOldUdaData{
      std::string udaName;
      std::string cellTypeName;
      std::string temperatureName;
      std::string abskgName;
      unsigned int timestep;
      int matl;
    };
    
    useOldUdaData* d_old_uda{nullptr};
  };

} // namespace Uintah

#endif
