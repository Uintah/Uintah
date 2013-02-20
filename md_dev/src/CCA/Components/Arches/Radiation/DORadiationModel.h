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

//----- DORadiationModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_DORadiationModel_h
#define Uintah_Component_Arches_DORadiationModel_h

/***************************************************************************
CLASS
    DORadiationModel
       Sets up the DORadiationModel
       
GENERAL INFORMATION
    DORadiationModel.h - Declaration of DORadiationModel class

    Author:Gautham Krishnamoorthy (gautham@crsim.utah.edu) 
           Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 06-18-2002

    C-SAFE
    
    
***************************************************************************/
//#include <CCA/Components/Arches/Radiation/RadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>

namespace Uintah {
  class ArchesLabel;
  class BoundaryCondition;

class DORadiationModel{

public:

      RadiationSolver* d_linearSolver;

      DORadiationModel(const ArchesLabel* label,
                       const MPMArchesLabel* MAlab,
                       BoundaryCondition* bndry_cond, 
                       const ProcessorGroup* myworld);


      virtual ~DORadiationModel();


      virtual void problemSetup(ProblemSpecP& params, bool stand_alone_src );

      virtual void computeRadiationProps(const ProcessorGroup* pc,
                                         const Patch* patch,
                                         CellInformation* cellinfo,
                                         ArchesVariables* vars,
                                         ArchesConstVariables* constvars);
      
      virtual void boundarycondition(const ProcessorGroup* pc,
                                     const Patch* patch,
                                     CellInformation* cellinfo, 
                                     ArchesVariables* vars,
                                     ArchesConstVariables* constvars);

      virtual void boundarycondition_new(const ProcessorGroup* pc,
                                         const Patch* patch,
                                         CellInformation* cellinfo, 
                                         ArchesVariables* vars,
                                         ArchesConstVariables* constvars);

      virtual void intensitysolve(const ProcessorGroup* pc,
                                  const Patch* patch,
                                  CellInformation* cellinfo, 
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars, 
                                  int wall_type);

      virtual void intensitysolve_new(const ProcessorGroup* pc,
                                  const Patch* patch,
                                  CellInformation* cellinfo, 
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars, 
                                  int wall_type);

      //__________________________________
      //  This task is called by enthalpy solver
      //  and it computes div_q
      virtual void sched_computeSource( const LevelP& level, 
                                SchedulerP& sched, 
                                const MaterialSet* matls,
                                const TimeIntegratorLabel* timelabels,
                                const bool isFirstIntegrationStep );
                                       
      virtual void computeSource( const ProcessorGroup* pc, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw,
                          const TimeIntegratorLabel* timelabels,  
                          bool isFirstIntegrationStep );

protected: 

      /// For other analytical properties.
      class PropertyCalculatorBase { 

        public: 
          PropertyCalculatorBase(); 
          virtual ~PropertyCalculatorBase(); 

          virtual bool problemSetup( const ProblemSpecP& db )=0; 
          virtual void computeProps( const Patch* patch, CCVariable<double>& abskg )=0;  // for now only assume abskg
      };

      //__________________________________
      //
      class ConstantProperties : public PropertyCalculatorBase  { 

        public: 
          ConstantProperties();
          ~ConstantProperties();

          bool problemSetup( const ProblemSpecP& db ) {
              
            ProblemSpecP db_prop = db; 
            db_prop->getWithDefault("abskg",_value,1.0); 
            
            bool property_on = true; 

            return property_on; 
          };

          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 
            abskg.initialize(_value); 
          }; 

        private: 
          double _value; 
      }; 
      //__________________________________
      //
      class  BurnsChriston : public PropertyCalculatorBase  { 

        public: 
          BurnsChriston();
          ~BurnsChriston();

          bool problemSetup( const ProblemSpecP& db ) { 
            ProblemSpecP db_prop = db; 
            db_prop->require("grid",grid);
             
            bool property_on = true; 
            return property_on; 
          };

          void computeProps( const Patch* patch, CCVariable<double>& abskg ){ 

            Vector Dx = patch->dCell(); 

            for (CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter){ 
              IntVector c = *iter; 
              std::cout << abskg[c] << std::endl;
              abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (grid.x() - 1.0) /2.0) * Dx[0]) )
                              * ( 1.0 - 2.0 * fabs( ( c[1] - (grid.y() - 1.0) /2.0) * Dx[1]) )
                              * ( 1.0 - 2.0 * fabs( ( c[2] - (grid.z() - 1.0) /2.0) * Dx[2]) ) 
                              + 0.1;
            } 
          }; 

        private: 
          double _value; 
          IntVector grid; 
      }; 

private:

      void computeOpticalLength();
      double d_opl; // optical length
      const ArchesLabel*    d_lab;
      const MPMArchesLabel* d_MAlab;
      BoundaryCondition* d_boundaryCondition;
      const ProcessorGroup* d_myworld;
      const PatchSet* d_perproc_patches;
      
      int  d_radCalcFreq;
      bool d_radRKsteps;
      bool d_radImpsteps;
      
      int d_sn, d_totalOrds; // totalOrdinates = sn*(sn+2)

      void computeOrdinatesOPL();
      
      int d_lambda;
      int ffield;
      bool lradcal, lwsgg, lplanckmean, lpatchmean;
      //not clear if these work so forcing them to be switched off: 
      bool lprobone, lprobtwo, lprobthree; 

      double d_wall_abskg; 
      double d_intrusion_abskg; 

      bool d_do_const_wall_T;
      double d_wall_temperature; 

      OffsetArray1<double> fraction;
      OffsetArray1<double> fractiontwo;

      OffsetArray1<double> oxi;
      OffsetArray1<double> omu;
      OffsetArray1<double> oeta;
      OffsetArray1<double> wt;

      OffsetArray1<double> rgamma;
      OffsetArray1<double> sd15;
      OffsetArray1<double> sd;
      OffsetArray1<double> sd7;
      OffsetArray1<double> sd3;

      OffsetArray1<double> srcbm;
      OffsetArray1<double> srcpone;
      OffsetArray1<double> qfluxbbm;

      bool d_use_abskp;
      const VarLabel* d_abskpLabel;


}; // end class RadiationModel

} // end namespace Uintah

#endif





