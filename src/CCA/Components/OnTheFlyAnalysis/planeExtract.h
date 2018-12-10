/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_planeExtract_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_planeExtract_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <map>
#include <vector>

namespace Uintah {
  

/**************************************

CLASS
   planeExtract
   
GENERAL INFORMATION

   planeExtract.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   planeExtract

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class planeExtract : public AnalysisModule {
  public:
    planeExtract(const ProcessorGroup* myworld,
                 const MaterialManagerP materialManager,
                 const ProblemSpecP& module_spec);
    
    planeExtract();
                    
    virtual ~planeExtract();
   
    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);
                              
    virtual void outputProblemSpec(ProblemSpecP& ps){};
                              
    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);
                                    
    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level){};
                                    
    virtual void restartInitialize();
                                    
    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);
   
    virtual void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                    const LevelP& level) {};
                                      
  private:

    enum PlaneType {XY=0, XZ=1, YZ=2, NONE=-9};
    
    void initialize( const ProcessorGroup*, 
                     const PatchSubset* patches,
                     const MaterialSubset*,
                     DataWarehouse*,
                     DataWarehouse* new_dw );
                    
    void doAnalysis( const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset*,
                     DataWarehouse*,
                     DataWarehouse* new_dw );
                    
    void createFile( const std::string& filename,
                     const VarLabel* varLabel,
                     const int matl,
                     FILE*& fp );
    
    void createDirectory( std::string& planeName,
                          std::string& timestep,
                          const double now,
                          std::string& levelIndex );
                          
    template <class Tvar>      /* double */
    void writeDataD( DataWarehouse*  new_dw,
                     const VarLabel* varLabel,
                     const int       indx,
                     const Patch*    patch,
                     const Vector&   offset,
                     CellIterator    iter,
                     FILE*     fp );
                     
    template <class Tvar>     /* Vector */
    void writeDataV( DataWarehouse*  new_dw,
                     const VarLabel* varLabel,
                     const int       indx,
                     const Patch*    patch,
                     const Vector&   offset,
                     CellIterator    iter,
                     FILE*     fp );
                     
    template <class Tvar>     /* integer */   
    void writeDataI( DataWarehouse*  new_dw,
                     const VarLabel* varLabel,
                     const int       indx,
                     const Patch*    patch,
                     const Vector&   offset,
                     CellIterator    iter,
                     FILE*     fp );
                     
    template <class Tvar>     /* Stencil7 */
    void writeDataS7( DataWarehouse*  new_dw,
                      const VarLabel* varLabel,
                      const int       indx,
                      const Patch*    patch,
                      const Vector&   offset,
                      CellIterator    iter,
                      FILE*     fp );
                      
    CellIterator getIterator( const Uintah::TypeDescription* td, 
                              const Patch* patch,
                              const IntVector& start_idx,
                              const IntVector& end_idx  );
                                         
    inline bool containsCellInclusive( const IntVector& low, 
                                       const IntVector& high, 
                                       const IntVector& cell,
                                       const PlaneType plane );
                                       
    // general labels
    class planeExtractLabel {
    public:
      VarLabel* lastWriteTimeLabel;
      VarLabel* fileVarsStructLabel;
    };
    
    planeExtractLabel* ps_lb;
   
    struct plane{
      std::string  name;
      Point   startPt;
      Point   endPt;
      double  startTime;
      double  stopTime;
      PlaneType planeType;
      
    };
       
    //__________________________________
    // global constants
    double d_writeFreq; 
    double d_startTime;
    double d_stopTime;
    
    std::vector<VarLabel*> d_varLabels;
    std::vector<int> d_varMatl;
    
    std::vector<plane*>   d_planes;
    std::set<std::string> d_isDirCreated;
    
    MaterialSet*     d_matl_set;
    MaterialSubset* d_zero_matl;  
  };
}

#endif
