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


#ifndef Uintah_Components_Arches_BoundaryCondition_h
#define Uintah_Components_Arches_BoundaryCondition_h


#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/LevelP.h>

#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include   <vector>

#include <CCA/Components/Arches/DigitalFilter/DigitalFilterInlet.h>

/**************************************
  CLASS
  BoundaryCondition

  Class BoundaryCondition applies boundary conditions
  at physical boundaries. For boundary cell types it
  modifies stencil coefficients and source terms.

  GENERAL INFORMATION
  BoundaryCondition.h - declaration of the class

Author: Rajesh Rawat (rawat@crsim.utah.edu)
Author of current BC formulation: Stanislav Borodai (borodai@crsim.utah.edu)

Creation Date:   Mar 1, 2000

C-SAFE 


KEYWORDS


DESCRIPTION
Class BoundaryCondition applies boundary conditions
at physical boundaries. For boundary cell types it
modifies stencil coefficients and source terms. 

WARNING
none
 ****************************************/

namespace Uintah {

  using namespace SCIRun;
  class ArchesVariables;
  class ArchesConstVariables;
  class CellInformation;
  class VarLabel;
  class PhysicalConstants;
  class Properties;
  class Stream;
  class InletStream;
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
  class DataWarehouse;
  class TimeIntegratorLabel;
  class IntrusionBC;
  class BoundaryCondition_new; 

  class BoundaryCondition {

    public:

      //** WARNING: This needs to be duplicated in BoundaryCond_new.h for now until BoundaryCondition goes away **//
      enum BC_TYPE { VELOCITY_INLET, MASSFLOW_INLET, VELOCITY_FILE, MASSFLOW_FILE, PRESSURE, OUTLET, WALL, MMWALL, INTRUSION, SWIRL, TURBULENT_INLET }; 
      enum DIRECTION { CENTER, EAST, WEST, NORTH, SOUTH, TOP, BOTTOM }; 

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a BoundaryCondition.
      // PRECONDITIONS
      // POSTCONDITIONS
      // Default constructor.
      BoundaryCondition();

      typedef std::map<std::string, constCCVariable<double> > HelperMap; 
      typedef std::vector<string> HelperVec;  
                        
                        /** @brief Stuct for hold face centered offsets relative to the cell centered boundary index. */
                        struct FaceOffSets { 
                                // Locations are determined by: 
                                //   (i,j,k) location = boundary_index - offset;
                                
                                public: 
                                        IntVector io;           ///< Interior cell offset 
                                        IntVector bo;     ///< Boundary cell offset 
                                        IntVector eo;           ///< Extra cell offset

                                  int sign;         ///< Sign of the normal for the face (ie, -1 for minus faces and +1 for plus faces )

                                        double dx;                              ///< cell size in the dimension of face normal (ie, distance between cell centers)

                        };

                 inline const FaceOffSets getFaceOffsets( const IntVector& face_normal, const Patch::FaceType face, const Vector Dx ){  

                         FaceOffSets offsets; 
                         offsets.io = IntVector(0,0,0);
                         offsets.bo = IntVector(0,0,0); 
                         offsets.eo = IntVector(0,0,0); 

                         if ( face == Patch::xminus || face == Patch::yminus || face == Patch::zminus ) { 

                                 offsets.bo = face_normal; 
                                 offsets.sign = -1; 

                         } else { 

                                 offsets.io = face_normal; 
                                 offsets.eo = IntVector( -1*face_normal.x(), -1*face_normal.y(), -1*face_normal.z() ); 
                                 offsets.sign = +1; 

                         } 

                         if ( face == Patch::xminus || face == Patch::xplus) { 
                                 offsets.dx = Dx.x(); 
                         } else if ( face == Patch::yminus || face == Patch::yplus ) { 
                                 offsets.dx = Dx.y(); 
                         } else { 
                                 offsets.dx = Dx.z(); 
                         } 

                         return offsets; 

                 };

     inline IntrusionBC* get_intrusion_ref(){ 
       return _intrusionBC; 
     }; 

     inline bool is_using_new_intrusion(){ 
       return _using_new_intrusion; 
     }; 

      inline bool typeMatch( BC_TYPE check_type, std::vector<BC_TYPE >& type_list ){ 

        bool found_match = false; 

        for ( std::vector<BC_TYPE>::iterator iter = type_list.begin(); iter != type_list.end(); ++iter ) { 
          if ( *iter == check_type ) {
            found_match = true; 
            return found_match; 
          } 
        } 

        return found_match; 
      };

     void sched_cellTypeInit__NEW(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);

     void cellTypeInit__NEW(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw);

     void sched_computeBCArea__NEW(SchedulerP& sched,
                                      const LevelP& level, 
                                      const PatchSet* patches,
                                      const MaterialSet* matls);
     void computeBCArea__NEW(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw, 
                                const IntVector lo, 
                                const IntVector hi);

     void sched_setupBCInletVelocities__NEW(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);

     void setupBCInletVelocities__NEW(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw);

     void sched_setInitProfile__NEW(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);

     void setInitProfile__NEW(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse*,
                              DataWarehouse* new_dw);

      void setVelFromExtraValue__NEW( const Patch* patch, const Patch::FaceType& face, 
        SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
        constCCVariable<double>& density, 
        Iterator bound_ptr, Vector value );

      void setVel__NEW( const Patch* patch, const Patch::FaceType& face, 
        SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel, 
        constCCVariable<double>& density, 
        Iterator bound_iter, Vector value );
    
      void setTurbInlet( const Patch* patch, const Patch::FaceType& face, 
                         SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel, 
                         constCCVariable<double>& density, 
                         Iterator bound_iter, DigitalFilterInlet * TurbIn );

      template<class d0T, class d1T, class d2T>
      void setSwirl( const Patch* patch, const Patch::FaceType& face, 
        d0T& uVel, d1T& vVel, d2T& wVel,
        constCCVariable<double>& density, 
        Iterator bound_ptr, Vector value, 
        double swirl_no, Vector swirl_cent );

      void setVelFromInput__NEW( const Patch* patch, const Patch::FaceType& face, 
        SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
        Iterator bound_iter, std::string file_name );

      void setEnthalpy__NEW( const Patch* patch, const Patch::FaceType& face, 
        CCVariable<double>& enthalpy, HelperMap ivGridVarMap, HelperVec ivNames,
        Iterator bound_ptr );

      void setEnthalpyFromInput__NEW( const Patch* patch, const Patch::FaceType& face, 
        CCVariable<double>& enthalpy, HelperMap ivGridVarMap, HelperVec ivNames, Iterator bound_ptr );

      void velocityOutletPressureBC__NEW( const Patch* patch, 
                                                int  matl_index, 
                                          SFCXVariable<double>& uvel, 
                                          SFCYVariable<double>& vvel, 
                                          SFCZVariable<double>& wvel, 
                                          constSFCXVariable<double>& old_uvel, 
                                          constSFCYVariable<double>& old_vvel, 
                                          constSFCZVariable<double>& old_wvel );

      template <class velType>
      void delPForOutletPressure__NEW( const Patch* patch, 
                                       int  matl_index, 
                                       double dt, 
                                       Patch::FaceType mface,
                                       Patch::FaceType pface, 
                                       velType& vel, 
                                       constCCVariable<double>& P,
                                       constCCVariable<double>& density );

      void sched_setPrefill__NEW( SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls);

      void setPrefill__NEW( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse* new_dw );


      template <class stencilType> 
      void zeroStencilDirection( const Patch* patch, 
                                 const int  matl_index, 
                                 const int sign, 
                                 stencilType& A, 
                                 std::vector<BC_TYPE>& types );
      
      template <class varType> void
      zeroGradientBC( const Patch* patch, 
                        const int  matl_index, 
                        varType& phi, 
                        std::vector<BC_TYPE>& types );

      std::map<IntVector, double>
      readInputFile__NEW( std::string );

      void sched_setupNewIntrusions(SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls); 

      void sched_setupNewIntrusionCellType( SchedulerP& sched, 
        const PatchSet* patches, const MaterialSet* matls, const bool doing_restart );

      void setHattedIntrusionVelocity( const Patch* p,
                                       SFCXVariable<double>& u, 
                                       SFCYVariable<double>& v, 
                                       SFCZVariable<double>& w, 
                                       constCCVariable<double>& density );

      void sched_setIntrusionDensity( SchedulerP& sched, const PatchSet* patches, const MaterialSet* matls );

      void setIntrusionDensity( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw);

      ////////////////////////////////////////////////////////////////////////
      // BoundaryCondition constructor used in  PSE
      BoundaryCondition(const ArchesLabel* label, 
          const MPMArchesLabel* MAlb,
          PhysicalConstants* phys_const, 
          Properties* props,
          bool calcReactScalar, 
          bool calcEnthalpy, 
          bool calcVariance);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~BoundaryCondition();

      // GROUP: Problem Steup:
      ////////////////////////////////////////////////////////////////////////
      // Details here
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Access functions

      bool getWallBC() { 
        return d_wallBoundary; 
      }

      bool getInletBC() { 
        return d_inletBoundary; 
      }

      bool getPressureBC() { 
        return d_pressureBoundary; 
      }

      bool getOutletBC() { 
        return d_outletBoundary; 
      }

      bool anyArchesPhysicalBC() { 
        return ((d_wallBoundary)||(d_inletBoundary)||(d_pressureBoundary)||(d_outletBoundary)); 
      }

      ////////////////////////////////////////////////////////////////////////
      // Get the number of inlets (primary + secondary)
      int getNumInlets() { 
        return d_numInlets; 
      }

      ////////////////////////////////////////////////////////////////////////
      // mm Wall boundary ID
      int getMMWallId() const {
        if ( d_use_new_bcs ) {
          return INTRUSION; 
        } else { 
          return d_mmWallID; 
        }
      }

      ////////////////////////////////////////////////////////////////////////
      // flowfield cell id
      inline int flowCellType() const {
        return d_flowfieldCellTypeVal;
        //return -1; 
      }

      ////////////////////////////////////////////////////////////////////////
      // Wall boundary ID
      inline int wallCellType() const { 
        if ( d_use_new_bcs ) { 
          return WALL; 
        } else { 
          return WALL; //wall_celltypeval; 
        } 
      }

      ////////////////////////////////////////////////////////////////////////
      // Pressure boundary ID
      inline int pressureCellType() const {
        if ( d_use_new_bcs ) { 
          return PRESSURE; 
        } else { 
          return PRESSURE; //pressure_celltypeval; 
        } 
      }

      ////////////////////////////////////////////////////////////////////////
      // Outlet boundary ID
      inline int outletCellType() const { 
        if ( d_use_new_bcs ) { 
          return OUTLET; 
        } else { 
          return OUTLET; //outlet_celltypeval; 
        } 
      }
      ////////////////////////////////////////////////////////////////////////
      // sets boolean for energy exchange between solid and fluid
      void setIfCalcEnergyExchange(bool calcEnergyExchange){
        d_calcEnergyExchange = calcEnergyExchange;
      }

      ////////////////////////////////////////////////////////////////////////
      // Access function for calcEnergyExchange (multimaterial)
      inline bool getIfCalcEnergyExchange() const{
        return d_calcEnergyExchange;
      }      

      ////////////////////////////////////////////////////////////////////////
      // sets boolean for fixing gas temperature in multimaterial cells
      void setIfFixTemp(bool fixTemp){
        d_fixTemp = fixTemp;
      }

      ////////////////////////////////////////////////////////////////////////
      // Access function for d_fixTemp (multimaterial)
      inline bool getIfFixTemp() const{
        return d_fixTemp;
      }      

      ////////////////////////////////////////////////////////////////////////
      // sets boolean for cut cells
      void setCutCells(bool cutCells){
        d_cutCells = cutCells;
      }

      ////////////////////////////////////////////////////////////////////////
      // Access function for d_cutCells (multimaterial)
      inline bool getCutCells() const{
        return d_cutCells;
      }      

      // GROUP:  Schedule tasks :
      ////////////////////////////////////////////////////////////////////////
      // Initialize cell types
      void sched_cellTypeInit(SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls);

      void sched_computeInletAreaBCSource(SchedulerP& sched, 
          const PatchSet* patches,
          const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Initialize inlet area
      // Details here
      void sched_calculateArea(SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Schedule Computation of Pressure boundary conditions terms. 
      void sched_computePressureBC(SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Schedule Set Profile BCS
      // initializes velocities, scalars and properties at the bndry
      // assigns flat velocity profiles for primary and secondary inlets
      // Also sets flat profiles for density
      // ** WARNING ** Properties profile not done yet
      void sched_setProfile(SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls);

      void sched_Prefill(SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls);

      void sched_initInletBC(SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls);


      template<class V, class T> void
        copy_stencil7(DataWarehouse* new_dw,
            const Patch* patch,
            const string& whichWay,
            CellIterator iter,
            V& A,  T& AP, T& AE, T& AW,
            T& AN, T& AS, T& AT, T& AB);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multimaterial wall cell types
      void sched_mmWallCellTypeInit( SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls, 
          bool fixCellType);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multimaterial wall cell types for first time step
      void sched_mmWallCellTypeInit_first( SchedulerP&, 
          const PatchSet* patches,
          const MaterialSet* matls);

      // GROUP:  Actual Computations :
      ////////////////////////////////////////////////////////////////////////
      // Initialize celltyping
      void cellTypeInit(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);
      ////////////////////////////////////////////////////////////////////////
      // computing inlet areas
      // Details here
      void computeInletFlowArea(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      void computeInletAreaBCSource(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset*,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute velocity BC terms
      void velocityBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute mms velocity BC terms
      void mmsvelocityBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          double time_shift,
          double dt);

      void mmsscalarBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          double time_shift,
          double dt);

      /** @brief Applies boundary conditions to A matrix for boundary conditions */
      void pressureBC(const Patch* patch,
          const int matl_index, 
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute scalar BC terms
      void scalarBC(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void scalarBC__new(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);


      ////////////////////////////////////////////////////////////////////////
      // Initialize multi-material wall celltyping and void fraction 
      // calculation
      void mmWallCellTypeInit(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw,
          bool fixCellType);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multi-material wall celltyping and void fraction 
      // calculation for first time step
      void mmWallCellTypeInit_first(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);
      // for computing intrusion bc's
      void intrusionTemperatureBC(const Patch* patch,
          constCCVariable<int>& cellType,
          CCVariable<double>& temperature);

      void mmWallTemperatureBC(const Patch* patch,
          constCCVariable<int>& cellType,
          constCCVariable<double> solidTemp,
          CCVariable<double>& temperature,
          bool d_energyEx);

      void calculateIntrusionVel(const Patch* patch,
          int index,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);


      void intrusionMomExchangeBC(const Patch* patch,
          int index, CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void intrusionEnergyExBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void intrusionScalarBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void intrusionEnthalpyBC(const Patch* patch, 
          double delta_t,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      // compute multimaterial wall bc
      void mmvelocityBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void mmpressureBC(DataWarehouse* new_dw,
          const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);
      // applies multimaterial bc's for scalars and pressure
      void mmscalarWallBC( const Patch* patch,
          CellInformation*, 
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      // applies multimaterial bc's for enthalpy
      void mmEnthalpyWallBC( const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Calculate uhat for multimaterial case (only for nonintrusion cells)
      void calculateVelRhoHat_mm(const Patch* patch,
          double delta_t,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void calculateVelocityPred_mm(const Patch* patch,
          double delta_t,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void scalarLisolve_mm(const Patch*,
          double delta_t,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          CellInformation* cellinfo);

      void enthalpyLisolve_mm(const Patch*,
          double delta_t,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          CellInformation* cellinfo);
      // New boundary conditions
      void scalarOutletPressureBC(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void velRhoHatInletBC(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          const int matl_index, 
          double time_shift);

      void velRhoHatOutletPressureBC( const Patch* patch,
                                      SFCXVariable<double>& uvel, 
                                      SFCYVariable<double>& vvel, 
                                      SFCZVariable<double>& wvel, 
                                      constSFCXVariable<double>& old_uvel, 
                                      constSFCYVariable<double>& old_vvel, 
                                      constSFCZVariable<double>& old_wvel, 
                                      constCCVariable<int>& cellType );

      void velocityOutletPressureTangentBC(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void addPresGradVelocityOutletPressureBC(const Patch* patch,
          CellInformation* cellinfo,
          const double delta_t,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void sched_getFlowINOUT(SchedulerP& sched,
          const PatchSet* patches,
          const MaterialSet* matls,
          const TimeIntegratorLabel* timelabels);

      void sched_correctVelocityOutletBC(SchedulerP& sched,
          const PatchSet* patches,
          const MaterialSet* matls,
          const TimeIntegratorLabel* timelabels);

      void sched_getScalarFlowRate(SchedulerP& sched,
          const PatchSet* patches,
          const MaterialSet* matls);

      void sched_getScalarEfficiency(SchedulerP& sched,
          const PatchSet* patches,
          const MaterialSet* matls);

      void sched_setInletFlowRates(SchedulerP& sched,
          const PatchSet* patches,
          const MaterialSet* matls);

      void mmsuVelocityBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          double time_shift,
          double dt);

      void mmsvVelocityBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          double time_shift,
          double dt);

      void mmswVelocityBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars,
          double time_shift,
          double dt);

      inline void setMMS(bool doMMS) {
        d_doMMS=doMMS;
      }
      inline bool getMMS() const {
        return d_doMMS;
      }

      //boundary source term methods
      void sched_computeScalarSourceTerm(SchedulerP& sched,
          const PatchSet* patches,
          const MaterialSet* matls);

      void sched_computeMomSourceTerm(SchedulerP& sched,
          const PatchSet* patches,
          const MaterialSet* matls);


      void sched_bcdummySolve( SchedulerP& sched, 
          const PatchSet* patches, 
          const MaterialSet* matls );

      void sched_setAreaFraction(SchedulerP& sched, 
          const PatchSet* patches, 
          const MaterialSet* matls );

      /** @brief The struct to hold all needed information for each boundary spec */
      struct BCInfo { 
  
        BC_TYPE type; 

        // Common: 
        //int id; 
        std::string name; 

        // Inlets: 
        Vector velocity; 
        double mass_flow_rate;
        std::string filename; 
        std::map<IntVector, double> file_input; 
        double swirl_no; 
        Vector swirl_cent; 

        // State: 
        double enthalpy; 
        double density; 

        // Varlabels: 
        const VarLabel* total_area_label; 
        
        DigitalFilterInlet * TurbIn;

      };

      void printBCInfo(){ 

        for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
            bc_iter != d_bc_information.end(); bc_iter++){

            std::cout << "/---------------------------------------------------------------/\n" << std::endl;
            std::cout << "Boundary type is: " << bc_iter->second.type << std::endl; 
            std::cout << "name: " << bc_iter->second.name << std::endl; 
            std::cout << "velocity components: " << bc_iter->second.velocity[0] << ", " << bc_iter->second.velocity[1] << ", " << bc_iter->second.velocity[2] << std::endl; 
            std::cout << "mass flow rate: " << bc_iter->second.mass_flow_rate << std::endl;
            std::cout << "file name: " << bc_iter->second.filename << std::endl;
            std::cout << "swirl no: " << bc_iter->second.swirl_no << " and centroid = " << bc_iter->second.swirl_cent << endl;
            std::cout << "enthalpy: " << bc_iter->second.enthalpy << std::endl;
            std::cout << "density: " << bc_iter->second.density << std::endl;
            std::cout << "area: " << bc_iter->second.total_area_label << std::endl;
            std::cout << "/---------------------------------------------------------------/\n" << std::endl;

        }
      }; 

      typedef std::map<BC_TYPE, std::string> BCNameMap;
      typedef std::map<int, BCInfo>      BCInfoMap;

      /** @brief Using the new BC mechanism? */
      inline bool isUsingNewBC(){ return d_use_new_bcs; }; 

      /** @brief Interface to the intrusion temperature method */ 
      void sched_setIntrusionTemperature( SchedulerP& sched, 
                                          const PatchSet* patches,
                                          const MaterialSet* matls );

      BCInfoMap d_bc_information;                           ///< Contains information about each boundary condition spec. (from UPS)

    private:

      /** @brief Setup new boundary conditions specified under the <Grid><BoundaryCondition> section */
      void setupBCs( ProblemSpecP& db ); 

      BCNameMap d_bc_type_to_string;                        ///< Matches the BC integer ID with the string name
      bool d_use_new_bcs;                                   ///< Turn on/off the new BC mech. 
      std::map<std::string, std::vector<GeometryPieceP> > d_prefill_map;  ///< Contains inlet name/geometry piece pairing

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute u velocity BC terms
      void intrusionuVelocityBC(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);


      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute v velocity BC terms
      void intrusionvVelocityBC(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute w velocity BC terms
      void intrusionwVelocityBC(const Patch* patch,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);


      void intrusionuVelMomExBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void intrusionvVelMomExBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);

      void intrusionwVelMomExBC(const Patch* patch,
          CellInformation* cellinfo,
          ArchesVariables* vars,
          ArchesConstVariables* constvars);


      ////////////////////////////////////////////////////////////////////////
      // Actually calculate pressure bcs
      void computePressureBC(const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);


      ////////////////////////////////////////////////////////////////////////
      // Actually set the velocity, density and props flat profile
      void setProfile(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      void Prefill(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);


      void initInletBC(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      // New boundary conditions

      void getFlowINOUT(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw,
          const TimeIntegratorLabel* timelabels);

      void correctVelocityOutletBC(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw,
          const TimeIntegratorLabel* timelabels);

      void getScalarFlowRate(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      void getScalarEfficiency(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      void getVariableFlowRate(const Patch* patch,
          CellInformation* cellinfo,
          ArchesConstVariables* constvars,
          constCCVariable<double> balance_var,
          double* varIN, 
          double* varOUT); 

      void setInletFlowRates(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);


      void bcdummySolve( const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset*,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      void setAreaFraction( const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset*,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw);

      void setFlatProfV( const Patch* patch, 
          SFCXVariable<double>& u, SFCYVariable<double>& v, SFCZVariable<double>& w, 
          const CCVariable<int>& cellType, const double area, const int inlet_type, 
          const double flow_rate, const double inlet_vel, const double density, 
          const bool xminus, const bool xplus, 
          const bool yminus, const bool yplus, 
          const bool zminus, const bool zplus, 
          double& actual_flow_rate ); 

      void setFlatProfS( const Patch* patch, 
          CCVariable<double>& scalar, 
          double value, 
          const CCVariable<int>& cellType, const double area, const int inlet_type, 
          const bool xminus, const bool xplus, 
          const bool yminus, const bool yplus, 
          const bool zminus, const bool zplus );

    private:

      /** @brief Fix a stencil direction to a specified value **/ 
      void fix_stencil_value( CCVariable<Stencil7>& stencil,  
          DIRECTION dir, double value, IntVector c ){

        switch ( dir ) { 
          case CENTER: 
            stencil[c].p = value; 
            break;
          case EAST: 
            stencil[c].e = value; 
            break; 
          case WEST: 
            stencil[c].w = value; 
            break; 
          case NORTH: 
            stencil[c].n = value; 
            break; 
          case SOUTH: 
            stencil[c].s = value; 
            break; 
          case TOP: 
            stencil[c].t = value; 
            break; 
          case BOTTOM: 
            stencil[c].b = value; 
            break; 
          default:
            break; 
        }
      };

      /** @brief Fix stencil to return a value ( v*x = c ), where x = solution variable **/ 
      void fix_value( CCVariable<Stencil7>& stencil, CCVariable<double>& su, CCVariable<double>& sp, 
          const double value, const double constant, IntVector c ){

        su[c] = constant * value; 
        sp[c] = -1.0 * constant;
        stencil[c].e = 0.0; 
        stencil[c].w = 0.0; 
        stencil[c].n = 0.0; 
        stencil[c].s = 0.0; 
        stencil[c].t = 0.0; 
        stencil[c].b = 0.0; 

        //note that stencil.p = sum(off_diagonals) - sp 

      };

      //-------------------------------------------------------------------
      // Flow Inlets
      //
      class FlowInlet {

        public:

          FlowInlet();
          FlowInlet(const FlowInlet& copy);
          FlowInlet(int cellID, bool calcVariance, bool reactingScalarSolve);
          ~FlowInlet();
          FlowInlet& operator=(const FlowInlet& copy);

          enum InletVelType { VEL_FLAT_PROFILE, VEL_FUNCTION, VEL_VECTOR, VEL_FILE_INPUT };
          enum InletScalarType { SCALAR_FLAT_PROFILE, SCALAR_FUNCTION, SCALAR_FILE_INPUT };

          InletVelType d_inletVelType; 
          InletScalarType d_inletScalarType; 

          int d_cellTypeID;          // define enum for cell type
          bool d_calcVariance;
          bool d_reactingScalarSolve;
          // inputs
          double flowRate;           
          double inletVel;           
          Vector d_velocity_vector; 
          double fcr;
          double fsr;
          int d_prefill_index;
          bool d_ramping_inlet_flowrate;
          bool d_prefill;
          InletStream streamMixturefraction;
          // calculated values
          Stream calcStream;
          // stores the geometry information, read from problem specs
          std::vector<GeometryPieceP> d_geomPiece;
          std::vector<GeometryPieceP> d_prefillGeomPiece;
          void problemSetup(ProblemSpecP& params);
          // reduction variable label to get area
          VarLabel* d_area_label;
          VarLabel* d_flowRate_label;
          string d_inlet_name; 
          double swirl_no; 
          Vector swirl_cent; 
          bool do_swirl; 
      };

      ////////////////////////////////////////////////////////////////////////
      // PressureInlet
      struct PressureInlet {
        int d_cellTypeID;
        bool d_calcVariance;
        bool d_reactingScalarSolve;
        InletStream streamMixturefraction;
        Stream calcStream;
        double area;
        // stores the geometry information, read from problem specs
        std::vector<GeometryPieceP> d_geomPiece;
        PressureInlet(int cellID, bool calcVariance, bool reactingScalarSolve);
        ~PressureInlet() {}
        void problemSetup(ProblemSpecP& params);
      };

      ////////////////////////////////////////////////////////////////////////
      // FlowOutlet
      struct FlowOutlet {
        int d_cellTypeID;
        bool d_calcVariance;
        bool d_reactingScalarSolve;
        InletStream streamMixturefraction;
        Stream calcStream;
        double area;
        // stores the geometry information, read from problem specs
        std::vector<GeometryPieceP> d_geomPiece;
        FlowOutlet(int cellID, bool calcVariance, bool reactingScalarSolve);
        ~FlowOutlet() {}
        void problemSetup(ProblemSpecP& params);
      };

      ////////////////////////////////////////////////////////////////////////
      // Wall Boundary
      struct WallBdry {
        int d_cellTypeID;
        double area;
        // stores the geometry information, read from problem specs
        std::vector<GeometryPieceP> d_geomPiece;
        WallBdry(int cellID);
        ~WallBdry() {}
        void problemSetup(ProblemSpecP& params);
      };

      void computeScalarSourceTerm(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset*,
          DataWarehouse*,
          DataWarehouse* new_dw);

      void computeMomSourceTerm(const ProcessorGroup*,
          const PatchSubset* patches,
          const MaterialSubset*,
          DataWarehouse*,
          DataWarehouse* new_dw);


      // Efficiency Variables
      struct EfficiencyInfo {
        const VarLabel* label; //efficiency label
        vector<std::string> species; 
        double fuel_ratio; 
        double air_ratio; 
        std::vector<string> which_inlets; //inlets needed for this calculation
      };

      struct SpeciesEfficiencyInfo {
        const VarLabel* flowRateLabel;
        double molWeightRatio; 
      };

      void insertIntoEffMap( std::string name, double fuel_ratio, double air_ratio, vector<std::string> species, vector<std::string> which_inlets ); 

      void insertIntoSpeciesMap ( std::string name, double mol_ratio );
    private:

      // input information
      typedef std::map<IntVector, double> CellToValue; 
      CellToValue _u_input; 
      CellToValue _v_input; 
      CellToValue _w_input; 

      // const VarLabel* inputs
      const ArchesLabel* d_lab;
      // for multimaterial
      const MPMArchesLabel* d_MAlab;
      int d_mmWallID;
      // cutoff for void fraction rqd to determine multimaterial wall
      double MM_CUTOFF_VOID_FRAC;
      bool d_calcEnergyExchange;
      bool d_fixTemp;
      bool d_cutCells;

      IntrusionBC* _intrusionBC; 
      bool _using_new_intrusion; 

      // used for calculating wall boundary conditions
      PhysicalConstants* d_physicalConsts;
      // used to get properties of different streams
      Properties* d_props;
      // mass flow
      double d_uvwout;
      double d_overallMB;
      // for reacting scalar
      bool d_reactingScalarSolve;
      // for enthalpy solve 
      bool d_enthalpySolve;
      bool d_calcVariance;
      // variable labels
      int d_flowfieldCellTypeVal;

      bool d_wallBoundary;
      WallBdry* d_wallBdry;

      bool d_inletBoundary;
      int d_numInlets;
      std::vector<FlowInlet* > d_flowInlets;

      bool d_pressureBoundary;
      PressureInlet* d_pressureBC;

      bool d_outletBoundary;
      FlowOutlet* d_outletBC;

      string d_mms;
      double d_airDensity, d_heDensity;
      Vector d_gravity;
      double d_viscosity;

      //linear mms
      double cu, cv, cw, cp, phi0;
      // sine mms
      double amp;

      double d_turbPrNo;
      bool d_doMMS;

      struct d_extraScalarBC {
        string d_scalar_name;
        double d_scalarBC_value;
        int d_BC_ID; 
      };
      vector<d_extraScalarBC*> d_extraScalarBCs; 

      typedef std::map<std::string, struct EfficiencyInfo> EfficiencyMap;
      EfficiencyMap d_effVars;

      typedef std::map<std::string, struct SpeciesEfficiencyInfo> SpeciesEffMap; // label string, molecular weight ratio 
      SpeciesEffMap d_speciesEffInfo;

      BoundaryCondition_new* d_newBC; 

      int index_map[3][3];


  inline int getNormal( Patch::FaceType face ) {        // This routine can be replaced with:
                                                        //  IntVector axes = patch->getFaceAxes(face);
                                                        //  int P_dir = axes[0];  // principal direction  --Todd
    int the_norm = -1; 

    switch ( face ) { 
      case Patch::xminus: 
        the_norm = 0;
        break; 
      case Patch::xplus: 
        the_norm = 0; 
        break; 
      case Patch::yminus: 
        the_norm = 1;
        break; 
      case Patch::yplus: 
        the_norm = 1; 
        break; 
      case Patch::zminus: 
        the_norm = 2;
        break; 
      case Patch::zplus: 
        the_norm = 2; 
        break; 
      default : 
        throw InvalidValue("In BoundaryCondition::getNormal, face not recognized.", __FILE__, __LINE__);
        break; 
    }
    return the_norm; 
  };

  }; // End of class BoundaryCondition


/** @brief Applies a zero gradient Neumann condition. This is a specialized case of a Neumann condition needed for old Arches code. */
template <class varType> void
BoundaryCondition::zeroGradientBC( const Patch* patch, 
                                   const int  matl_index, 
                                   varType& phi, 
                                   std::vector<BC_TYPE>& types )

{
  vector<Patch::FaceType>::const_iterator bf_iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

     if ( typeMatch( bc_iter->second.type, types ) ) { 

       for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ){

         Patch::FaceType face    = *bf_iter;
         IntVector insideCellDir = patch->faceDirection(face); 

         //get the number of children
         int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

         for (int child = 0; child < numChildren; child++){

           double bc_value = 0;
           Vector bc_v_value(0,0,0); 
           string bc_kind = "NotSet";
           Iterator bound_ptr;
           bool foundIterator = false; 

           if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ){ 
            foundIterator = 
              getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind); 
           } else { 
            foundIterator = 
              getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 
           } 

           if ( foundIterator ) {

             for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
               IntVector c = *bound_ptr; 
               IntVector c_int = *bound_ptr - insideCellDir; 

               phi[c] = phi[c_int]; 

             }
           }
         }
       }
     }
   }
}
/** @brief Zeroes out contribution to a stencil in a specified direction.  Also removes it from the diagonal, typically used for BCs */
template <class stencilType> void
BoundaryCondition::zeroStencilDirection( const Patch* patch, 
                                         const int  matl_index, 
                                         const int  sign, 
                                         stencilType& A, 
                                         std::vector<BC_TYPE>& types )

{
  vector<Patch::FaceType>::const_iterator bf_iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

     if ( typeMatch( bc_iter->second.type, types ) ) { 

       for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ){

         Patch::FaceType face    = *bf_iter;
         IntVector insideCellDir = patch->faceDirection(face); 

         //get the number of children
         int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

         for (int child = 0; child < numChildren; child++){

           double bc_value = 0;
           Vector bc_v_value(0,0,0); 
           string bc_kind = "NotSet";
           Iterator bound_ptr;
           bool foundIterator = false; 

           if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ){ 
            foundIterator = 
              getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind); 
           } else { 
            foundIterator = 
              getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 
           } 

           if ( foundIterator ) {

             for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
               IntVector c = *bound_ptr; 
               c -= insideCellDir; //because we are adjusting the interior stencil values. 

               A[c].p = A[c].p + ( sign * A[c][face] );
               A[c][face] = 0.0;

             }
           }
         }
       }
     }
   }
}

/** @brief Adds grad(P) to velocity on outlet or pressure boundaries */
template <class velType> void
BoundaryCondition::delPForOutletPressure__NEW( const Patch* patch, 
                                               int  matl_index, 
                                               double dt,
                                               Patch::FaceType mface, 
                                               Patch::FaceType pface, 
                                               velType& vel, 
                                               constCCVariable<double>& P,
                                               constCCVariable<double>& density )
{

  vector<Patch::FaceType>::const_iterator bf_iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell(); 

  for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

    if ( bc_iter->second.type == OUTLET || bc_iter->second.type == PRESSURE ) { 

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ){

        //get the face
        Patch::FaceType face    = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face); 

        if ( face == mface || face == pface ) { 

          //get the number of children
          int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

          for (int child = 0; child < numChildren; child++){

            double bc_value = 0;
            string bc_kind = "NotSet";
            Iterator bound_ptr;

            // No need to check for vector since this is an outlet or pressure
            bool foundIterator = 
              getIteratorBCValueBCKind( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 

            if ( foundIterator ) {

              const FaceOffSets FOS = getFaceOffsets( insideCellDir, face, Dx ); 

               for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                 IntVector c = *bound_ptr; 

                 double ave_density = 0.5 * ( density[c] + density[c - insideCellDir] ); 

                 double gradP = FOS.sign * 2.0 * dt * P[c - insideCellDir] / ( FOS.dx * ave_density ); 

                 vel[c - FOS.bo] += gradP; 

                 vel[c - FOS.eo] = vel[c - FOS.bo]; 

               }
            }
          }
        }
      }
    }
  }
}

} // End namespace Uintah

#endif  

