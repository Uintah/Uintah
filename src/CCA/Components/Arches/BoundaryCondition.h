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


#ifndef Uintah_Components_Arches_BoundaryCondition_h
#define Uintah_Components_Arches_BoundaryCondition_h


#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Task.h>

#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <vector>
#include <ostream>
#include <fstream>

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

class ArchesVariables;
class ArchesConstVariables;
class CellInformation;
class VarLabel;
class PhysicalConstants;
class Properties;
class TableLookup;
class ArchesLabel;
class MPMArchesLabel;
class ProcessorGroup;
class DataWarehouse;
class TimeIntegratorLabel;
class IntrusionBC;
class BoundaryCondition_new;
class WBCHelper;

class BoundaryCondition {

public:

//** WARNING: This needs to be duplicated in BoundaryCond_new.h for now until BoundaryCondition goes away **//
//** WARNING!!! ** //
enum BC_TYPE { VELOCITY_INLET, MASSFLOW_INLET,  VELOCITY_FILE, MASSFLOW_FILE, STABL, PRESSURE,
         OUTLET, NEUTRAL_OUTLET, WALL, MMWALL, INTRUSION, SWIRL, TURBULENT_INLET, PARTMASSFLOW_INLET};
//** END WARNING!!! **//


enum DIRECTION { CENTER, EAST, WEST, NORTH, SOUTH, TOP, BOTTOM };

// GROUP: Constructors:
////////////////////////////////////////////////////////////////////////
// Construct an instance of a BoundaryCondition.
// PRECONDITIONS
// POSTCONDITIONS
// Default constructor.
BoundaryCondition();

typedef std::map<std::string, constCCVariable<double> > HelperMap;
typedef std::vector<std::string> HelperVec;

/** @brief Stuct for hold face centered offsets relative to the cell centered boundary index. */
struct FaceOffSets {
  // Locations are determined by:
  //   (i,j,k) location = boundary_index - offset;

public:
  IntVector io;                                           ///< Interior cell offset
  IntVector bo;                                     ///< Boundary cell offset
  IntVector eo;                                           ///< Extra cell offset

  int sign;                                   ///< Sign of the normal for the face (ie, -1 for minus faces and +1 for plus faces )

  double dx;                                                              ///< cell size in the dimension of face normal (ie, distance between cell centers)

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

inline const std::map<int, IntrusionBC*> get_intrusion_ref(){
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

Vector getMaxIntrusionVelocity( const Level* level );

void prune_per_patch_bcinfo( SchedulerP& sched,
                             const LevelP& level,
                             WBCHelper* bcHelper );

void sched_cellTypeInit( SchedulerP& sched,
                         const LevelP& level,
                         const MaterialSet* matls);

void cellTypeInit( const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse*,
                   DataWarehouse* new_dw,
                   IntVector lo, IntVector hi);

void sched_setupBCInletVelocities( SchedulerP& sched,
                                   const LevelP& level,
                                   const MaterialSet* matls,
                                   bool doing_restart,
                                   bool doing_regrid);

void setupBCInletVelocities( const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse*,
                             DataWarehouse* new_dw,
                             bool doing_regrid);

void setupBCInletVelocitiesHack( const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw ){}

void sched_setInitProfile( SchedulerP& sched,
                           const LevelP& level,
                           const MaterialSet* matls);

void setInitProfile( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset*,
                     DataWarehouse*,
                     DataWarehouse* new_dw );

void sched_checkMomBCs( SchedulerP& sched,
                        const LevelP& level,
                        const MaterialSet* matls );

void checkMomBCs( const ProcessorGroup* pc,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw );

void setVelFromExtraValue( const Patch* patch, const Patch::FaceType& face,
                           SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                           constCCVariable<double>& density,
                           Iterator bound_ptr, Vector value );

void setVel( const Patch* patch, const Patch::FaceType& face,
             SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
             constCCVariable<double>& density,
             Iterator bound_iter, Vector value );


void setTurbInlet( const Patch* patch, const Patch::FaceType& face,
                   SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                   constCCVariable<double>& density,
                   Iterator bound_iter, DigitalFilterInlet * TurbIn,
                   const int timeStep,
                   const double simTime  );

template<class d0T, class d1T, class d2T>
void setSwirl( const Patch* patch, const Patch::FaceType& face,
               d0T& uVel, d1T& vVel, d2T& wVel,
               constCCVariable<double>& density,
               Iterator bound_ptr, Vector value,
               double swirl_no, Vector swirl_cent );

void setVelFromInput( const Patch* patch, const Patch::FaceType& face, std::string face_name,
                           SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                           Iterator bound_iter, std::string file_name );

void velocityOutletPressureBC( const Patch* patch,
                                    int matl_index,
                                    SFCXVariable<double>& uvel,
                                    SFCYVariable<double>& vvel,
                                    SFCZVariable<double>& wvel,
                                    constSFCXVariable<double>& old_uvel,
                                    constSFCYVariable<double>& old_vvel,
                                    constSFCZVariable<double>& old_wvel );

template <class velType>
void delPForOutletPressure( const Patch* patch,
                                 int matl_index,
                                 double dt,
                                 Patch::FaceType mface,
                                 Patch::FaceType pface,
                                 velType& vel,
                                 constCCVariable<double>& P,
                                 constCCVariable<double>& density );

template <class stencilType>
void zeroStencilDirection( const Patch* patch,
                           const int matl_index,
                           const int sign,
                           stencilType& A,
                           std::vector<BC_TYPE>& types );

template <class varType> void
zeroGradientBC( const Patch* patch,
                const int matl_index,
                varType& phi,
                std::vector<BC_TYPE>& types );

void sched_setupNewIntrusions( SchedulerP&,
                               const LevelP& level,
                               const MaterialSet* matls );

void sched_setupNewIntrusionCellType( SchedulerP& sched,
                                      const LevelP& level,
                                      const MaterialSet* matls,
                                      const bool doing_restart );

void sched_setIntrusionDensity( SchedulerP& sched,
                                const LevelP& level,
                                const MaterialSet* matls );

void setIntrusionDensity( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse*,
                          DataWarehouse* new_dw);
                          
void sched_computeAlphaG( SchedulerP& sched,
                          const LevelP& level,
                          const MaterialSet* matls,
                          const bool carry_forward );

// wall closure models:
void
sched_wallStressConstSmag( Task::WhichDW dw, Task* tsk );

void
wallStressConstSmag( const Patch* p,
                     DataWarehouse* dw,
                     const double wall_C_smag,
                     const int standoff,
                     constSFCXVariable<double>& uvel,
                     constSFCYVariable<double>& vvel,
                     constSFCZVariable<double>& wvel,
                     SFCXVariable<double>& Su,
                     SFCYVariable<double>& Sv,
                     SFCZVariable<double>& Sw,
                     constCCVariable<double>& rho,
                     constCCVariable<double>& eps );
void
wallStressMolecular( const Patch* p,
                     constSFCXVariable<double>& uvel,
                     constSFCYVariable<double>& vvel,
                     constSFCZVariable<double>& wvel,
                     SFCXVariable<double>& Su,
                     SFCYVariable<double>& Sv,
                     SFCZVariable<double>& Sw,
                     constCCVariable<double>& eps );
void wallStressLog( const Patch* patch,
                    ArchesVariables* vars,
                    ArchesConstVariables* const_vars,
                    constCCVariable<double>& volFraction );

inline void   newton_solver( const double&  , const double& , const double& , const double& ,double& );

void
wallStressDynSmag( const Patch* p,
                   const int standoff,
                   constCCVariable<double>& mu_t,
                   constSFCXVariable<double>& uvel,
                   constSFCYVariable<double>& vvel,
                   constSFCZVariable<double>& wvel,
                   SFCXVariable<double>& Su,
                   SFCYVariable<double>& Sv,
                   SFCZVariable<double>& Sw,
                   constCCVariable<double>& rho,
                   constCCVariable<double>& eps );

/** @brief Set the address for the BC helper created in the non-linear solver **/
void setBCHelper( std::map<int,WBCHelper*>* helper ){m_bcHelper = helper;}

/** @brief Copy the temperature into a radiation temperature for use later. Also forces BCs in extra cell **/
void sched_create_radiation_temperature( SchedulerP        & sched,
                                         const LevelP      & level,
                                         const MaterialSet * matls,
                                         const bool doing_restart,
                                         const bool use_old_dw );

/** @brief See sched_create_radiation_temperature **/
void create_radiation_temperatureHack( const ProcessorGroup* pc,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw){};

/** @brief See sched_create_radiation_temperature **/
void create_radiation_temperature( const ProcessorGroup* pc,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw,
                                   const bool use_old_dw);

/** @brief Add the intrusion momentum source to vel hats RHS **/
void
addIntrusionMomRHS( const Patch* patch,
                    constSFCXVariable<double>& uVel,
                    constSFCYVariable<double>& vVel,
                    constSFCZVariable<double>& wVel,
                    SFCXVariable<double>& usrc,
                    SFCYVariable<double>& vsrc,
                    SFCZVariable<double>& wsrc,
                    constCCVariable<double>& density );

/** @brief Add the intrusion mass source **/
void
addIntrusionMassRHS( const Patch* patch,
                     CCVariable<double>& mass_src );

BoundaryCondition(const ArchesLabel* label,
                  const MPMArchesLabel* MAlb,
                  PhysicalConstants* phys_const,
                  Properties* props,
                  TableLookup* table_lookup);


~BoundaryCondition();

void problemSetup( const ProblemSpecP& params, GridP& grid );

void set_bc_information( const LevelP& level );

////////////////////////////////////////////////////////////////////////
// mm Wall boundary ID
int getMMWallId() const {
  return INTRUSION;
}

////////////////////////////////////////////////////////////////////////
// Wall boundary ID
inline int wallCellType() const {
  return WALL;
}

////////////////////////////////////////////////////////////////////////
// Pressure boundary ID
inline int pressureCellType() const {
  return PRESSURE;
}

////////////////////////////////////////////////////////////////////////
// Outlet boundary ID
inline int outletCellType() const {
  return OUTLET;
}

////////////////////////////////////////////////////////////////////////
// sets boolean for energy exchange between solid and fluid
void setIfCalcEnergyExchange(bool calcEnergyExchange){
  d_calcEnergyExchange = calcEnergyExchange;
}

////////////////////////////////////////////////////////////////////////
// Access function for calcEnergyExchange (multimaterial)
inline bool getIfCalcEnergyExchange() const {
  return d_calcEnergyExchange;
}

////////////////////////////////////////////////////////////////////////
// sets boolean for fixing gas temperature in multimaterial cells
void setIfFixTemp(bool fixTemp){
  d_fixTemp = fixTemp;
}

////////////////////////////////////////////////////////////////////////
// Access function for d_fixTemp (multimaterial)
inline bool getIfFixTemp() const {
  return d_fixTemp;
}

////////////////////////////////////////////////////////////////////////
// sets boolean for cut cells
void setCutCells(bool cutCells){
  d_cutCells = cutCells;
}

////////////////////////////////////////////////////////////////////////
// Access function for d_cutCells (multimaterial)
inline bool getCutCells() const {
  return d_cutCells;
}

void sched_computeInletAreaBCSource(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls);

////////////////////////////////////////////////////////////////////////
// Schedule Computation of Pressure boundary conditions terms.
void sched_computePressureBC(SchedulerP&,
                             const PatchSet* patches,
                             const MaterialSet* matls);

template<class V, class T> void
copy_stencil7(DataWarehouse* new_dw,
              const Patch* patch,
              const std::string& whichWay,
              CellIterator iter,
              V& A,  T& AP, T& AE, T& AW,
              T& AN, T& AS, T& AT, T& AB);

void computeInletAreaBCSource(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

/** @brief Applies boundary conditions to A matrix for boundary conditions */
void pressureBC(const Patch* patch,
                const int matl_index,
                ArchesVariables* vars,
                ArchesConstVariables* constvars);

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
void wallVelocityBC(const Patch* patch,
                    CellInformation* cellinfo,
                    ArchesVariables* vars,
                    ArchesConstVariables* constvars);

void mmpressureBC(DataWarehouse* new_dw,
                  const Patch* patch,
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

void velRhoHatInletBC(const Patch* patch,
                      ArchesVariables* vars,
                      ArchesConstVariables* constvars,
                      const int matl_index,
                      const int timeStep,
                      const double simTime,
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

//boundary source term methods
void sched_computeScalarSourceTerm(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls);

void sched_computeMomSourceTerm(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls);


void sched_setAreaFraction(SchedulerP& sched,
                           const LevelP& level,
                           const MaterialSet* matls,
                           const int timesubstep,
                           const bool reinitialize );

/** @brief The struct to hold all needed information for each boundary spec */
struct BCInfo {

  BC_TYPE type;

  // Common:
  //int id;
  std::string name;
  std::string partName;
  std::string faceName;
  bool lHasPartMassFlow;
  Patch::FaceType face;

  // Inlets:
  Vector velocity;
  Vector partVelocity;
  Vector unitVector;
  Vector partUnitVector;
  double mass_flow_rate;
  std::string filename;
  std::map<IntVector, double> file_input;
  double swirl_no;
  Vector swirl_cent;

  //Stabilized Atmospheric BL
  double zo;
  double zh;
  double k;
  double kappa;
  double ustar;
  int dir_gravity;

  // State:
  double enthalpy;
  double density;
  double partDensity;

  DigitalFilterInlet * TurbIn;

  std::vector<double> vWeights;
  std::vector<std::vector<double> > vVelScalingConst;
  std::vector<std::vector<std::string> > vVelLabels;

};

void setStABL( const Patch* patch, const Patch::FaceType& face,
               SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
               BCInfo* bcinfo,
               Iterator bound_ptr  );


typedef std::map<BC_TYPE, std::string> BCNameMap;
typedef std::map<int, BCInfo>      BCInfoMap;

/** @brief Interface to the intrusion temperature method */
void sched_setIntrusionTemperature( SchedulerP& sched,
                                    const LevelP& level,
                                    const MaterialSet* matls );

std::map<int, BCInfoMap> d_bc_information; ///< Contains information about each boundary condition spec. (from UPS)

BoundaryCondition_new* getNewBoundaryCondition(){
  return d_newBC;
}                                                                           // needed by Arches:RMCRT

private:

bool m_has_boundaries{true};

std::map<const std::string, const VarLabel*> m_area_labels;
std::map<int,WBCHelper*>* m_bcHelper;
Uintah::ProblemSpecP m_arches_spec;

/** @brief Makes a label for the reduction of the surface area for this child **/
void create_new_area_label( const std::string name );

/** @brief Setup new boundary conditions specified under the <Grid><BoundaryCondition> section */
void setupBCs( ProblemSpecP db, const LevelP& level );

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

void setAreaFraction( const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset*,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const int timesubstep,
                      const bool reinitialize );


bool d_no_corner_recirc;

/** @brief A method for applying the outlet/pressure BC on minus faces. **/
template <class velType, class oldVelType> void
outletPressureMinus( IntVector insideCellDir,
                     Iterator bound_ptr,
                     double const sign, velType& vel,
                     oldVelType& old_vel );

/** @brief A method for applying the outlet/pressure BC on plus faces. **/
template <class velType, class oldVelType> void
outletPressurePlus(  IntVector insideCellDir,
                     Iterator bound_ptr,
                     double const sign, velType& vel,
                     oldVelType& old_vel );

/** @brief A method for applying the outlet/pressure BC on minus faces.
           This method has Stas' corner constraint. **/
template <class velType, class oldVelType> void
outletPressureMinus( IntVector insideCellDir,
                     Iterator bound_ptr,
                     IntVector idxLo, IntVector idxHi,
                     const int t1, const int t2,
                     double const sign, velType& vel,
                     oldVelType& old_vel,
                     const bool mF1, const bool pF1,
                     const bool mF2, const bool pF2 );

/** @brief A method for applying the outlet/pressure BC on plus faces.
           This method has Stas' corner constraint. **/
template <class velType, class oldVelType> void
outletPressurePlus( IntVector insideCellDir,
                    Iterator bound_ptr,
                    IntVector idxLo, IntVector idxHi,
                    const int t1, const int t2,
                    double const sign, velType& vel,
                    oldVelType& old_vel,
                    const bool mF1, const bool pF1,
                    const bool mF2, const bool pF2 );

/** @brief A method for applying the outlet BC du/dn=0 on minus faces. **/
template <class velType, class oldVelType> void
neutralOutleMinus( IntVector insideCellDir,
                   Iterator bound_ptr,
                   velType& vel,
                   oldVelType& old_vel );

/** @brief A method for applying the outlet BC du/dn=0 on plus faces. **/
template <class velType, class oldVelType> void
neutralOutletPlus(  IntVector insideCellDir,
                    Iterator bound_ptr,
                    velType& vel,
                    oldVelType& old_vel );

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

// input information
typedef std::map<IntVector, double> CellToValue;
struct FFInfo {
  CellToValue values;
  Vector relative_xyz;
  double dx;
  double dy;
  IntVector relative_ijk;
  std::string default_type;
  std::string name;
  double default_value;
};
typedef std::map<std::string, FFInfo> FaceToInput;

FaceToInput _u_input;
FaceToInput _v_input;
FaceToInput _w_input;

void readInputFile( std::string, BoundaryCondition::FFInfo& info, const int index );
std::vector<std::string> d_all_v_inlet_names;

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

bool d_check_inlet_obstructions;

bool d_ignore_invalid_celltype;

std::map<int, IntrusionBC*> _intrusionBC;
bool _using_new_intrusion;

// used for calculating wall boundary conditions
PhysicalConstants* d_physicalConsts;
// used to get properties of different streams
Properties* d_props;
TableLookup* d_table_lookup;
// mass flow
double d_uvwout;
double d_overallMB;

std::string d_mms;
double d_airDensity, d_heDensity;
Vector d_gravity;
double d_viscosity;

//linear mms
double cu, cv, cw, cp, phi0;
// sine mms
double amp;

double d_turbPrNo;
bool d_slip;
double d_csmag_wall;

struct d_extraScalarBC {
  std::string d_scalar_name;
  double d_scalarBC_value;
  int d_BC_ID;
};
std::vector<d_extraScalarBC*> d_extraScalarBCs;

BoundaryCondition_new* d_newBC;

int index_map[3][3];

const VarLabel* d_radiation_temperature_label;           // a copy of temperature from table with "forced" BC in the extra cell
const VarLabel* d_temperature_label;

inline int getNormal( Patch::FaceType face ) {          // This routine can be replaced with:
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
  default:
    throw InvalidValue("In BoundaryCondition::getNormal, face not recognized.", __FILE__, __LINE__);
    break;
  }
  return the_norm;
};

/** @brief Will identify a cell as a corner if it is such. Should only use
 *         this function for setup type operations.  Could move the xminus, etc.,
 *         determination outside the function for slightly better performance. **/
bool inline is_corner_cell( const Patch* patch, IntVector c,
                            IntVector lo, IntVector hi ){

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;

  bool is_corner = false;

  if ( xminus && yminus ) {
    if ( c[0] == lo.x()-1 ) {
      is_corner = true;
    }
  }

  if ( xminus && yplus ) {
    if ( c[0] == lo.x()-1 ) {
      is_corner = true;
    }
  }

  if ( xplus && yminus ) {
    if ( c[0] == hi.x() ) {
      is_corner = true;
    }
  }

  if ( xplus && yplus ) {
    if ( c[0] == hi.x() ) {
      is_corner = true;
    }
  }

  if ( xminus && zminus ) {
    if ( c[0] == lo.x()-1 ) {
      is_corner = true;
    }
  }

  if ( xminus && zplus ) {
    if ( c[0] == lo.x()-1 ) {
      is_corner = true;
    }
  }

  if ( xplus && zminus ) {
    if ( c[0] == hi.x() ) {
      is_corner = true;
    }
  }

  if ( xplus && zplus ) {
    if ( c[0] == hi.x() ) {
      is_corner = true;
    }
  }
  //---
  if ( yminus && zminus ) {
    if ( c[1] == lo.y()-1 ) {
      is_corner = true;
    }
  }

  if ( yminus && zplus ) {
    if ( c[1] == lo.y()-1 ) {
      is_corner = true;
    }
  }

  if ( yplus && zminus ) {
    if ( c[1] == hi.y() ) {
      is_corner = true;
    }
  }

  if ( yplus && zplus ) {
    if ( c[1] == hi.y() ) {
      is_corner = true;
    }
  }

  return is_corner;

}

// Pick a side based on UPS spec:
Patch::FaceType getFaceTypeFromUPS( ProblemSpecP db_face ){
  //Assuming that the Face node is coming into this function

  std::string face_type_str;

  //side check
  if ( db_face->findAttribute("side") ){
      db_face->getAttribute("side", face_type_str);
  }

  //circle check
  if ( db_face->findAttribute("circle") ){
    db_face->getAttribute("circle", face_type_str );
  }

  //annulus check
  if ( db_face->findAttribute("annulus") ){
    db_face->getAttribute("annulus", face_type_str );
  }

  //rectangle check
  if ( db_face->findAttribute("rectangle") ){
    db_face->getAttribute("rectangle", face_type_str );
  }

  //ellipse check
  if ( db_face->findAttribute("ellipse") ){
    db_face->getAttribute("ellipse", face_type_str );
  }

  //rectangulus check
  if ( db_face->findAttribute("rectangulus") ){
    db_face->getAttribute("rectangulus", face_type_str );
  }

  if ( face_type_str == "x-" ){
    return Patch::xminus;
  } else if ( face_type_str == "x+" ){
    return Patch::xplus;
  } else if ( face_type_str == "y-" ){
    return Patch::yminus;
  } else if ( face_type_str == "y+" ){
    return Patch::yplus;
  } else if ( face_type_str == "z-" ){
    return Patch::zminus;
  } else if ( face_type_str == "z+" ){
    return Patch::zplus;
  } else {
    return Patch::invalidFace;
  }

}

};   // End of class BoundaryCondition


/** @brief Applies a zero gradient Neumann condition. This is a specialized case of a Neumann condition needed for old Arches code. */
template <class varType> void
BoundaryCondition::zeroGradientBC( const Patch* patch,
                                   const int matl_index,
                                   varType& phi,
                                   std::vector<BC_TYPE>& types )

{
  std::vector<Patch::FaceType>::const_iterator bf_iter;
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  const Level* level = patch->getLevel();
  const int ilvl = level->getID();

  for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
        bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

    if ( typeMatch( bc_iter->second.type, types ) ) {

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ) {

        Patch::FaceType face    = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face);

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++) {

          double bc_value = 0;
          Vector bc_v_value(0,0,0);
          std::string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false;

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ) {
            foundIterator =
                    getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind);
          } else {
            foundIterator =
                    getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);
          }

          if ( foundIterator ) {

            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
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
                                         const int matl_index,
                                         const int sign,
                                         stencilType& A,
                                         std::vector<BC_TYPE>& types )

{
  std::vector<Patch::FaceType>::const_iterator bf_iter;
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  const Level* level = patch->getLevel();
  const int ilvl = level->getID();

  for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
        bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

    if ( typeMatch( bc_iter->second.type, types ) ) {

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ) {

        Patch::FaceType face    = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face);

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++) {

          double bc_value = 0;
          Vector bc_v_value(0,0,0);
          std::string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false;

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ) {
            foundIterator =
                    getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind);
          } else {
            foundIterator =
                    getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);
          }

          if ( foundIterator ) {

            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
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
BoundaryCondition::delPForOutletPressure( const Patch* patch,
                                          int matl_index,
                                          double dt,
                                          Patch::FaceType mface,
                                          Patch::FaceType pface,
                                          velType& vel,
                                          constCCVariable<double>& P,
                                          constCCVariable<double>& density )
{

  std::vector<Patch::FaceType>::const_iterator bf_iter;
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell();

  const Level* level = patch->getLevel();
  const int ilvl = level->getID();

  for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
        bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

    if ( bc_iter->second.type == OUTLET ||
         bc_iter->second.type == PRESSURE ||
         bc_iter->second.type == NEUTRAL_OUTLET ) {

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ) {

        //get the face
        Patch::FaceType face    = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face);

        if ( face == mface || face == pface ) {

          //get the number of children
          int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

          for (int child = 0; child < numChildren; child++) {

            double bc_value = 0;
            std::string bc_kind = "NotSet";
            Iterator bound_ptr;

            // No need to check for vector since this is an outlet or pressure
            bool foundIterator =
                    getIteratorBCValueBCKind( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);

            if ( foundIterator ) {

              const FaceOffSets FOS = getFaceOffsets( insideCellDir, face, Dx );

              for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

                IntVector c = *bound_ptr;

                double ave_density = 0.5 * ( density[c] + density[c - insideCellDir] );

                if ( ave_density > 1e-10 ) {
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
}

template <class velType, class oldVelType> void
BoundaryCondition::outletPressureMinus( IntVector insideCellDir,
                                        Iterator bound_ptr,
                                        double const sign, velType& vel,
                                        oldVelType& old_vel )
{

  double const negsmall = -1.0E-10;
  double const zero     = 0.0E0;
  for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

    IntVector c = *bound_ptr;
    IntVector cp  = c - insideCellDir;
    IntVector cpp = cp - insideCellDir;

    if ( sign * old_vel[cp] < negsmall ) {
      vel[cp] = vel[cpp];
    } else {
      vel[cp] = zero;
    }
    vel[c] = vel[cp];
  }
}
template <class velType, class oldVelType> void
BoundaryCondition::outletPressurePlus( IntVector insideCellDir,
                                       Iterator bound_ptr,
                                       double const sign, velType& vel,
                                       oldVelType& old_vel )
{

  double possmall =  1.0E-10;
  double zero     = 0.0E0;
  for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

    IntVector c = *bound_ptr;
    IntVector cp  = c - insideCellDir;
    IntVector cm  = c + insideCellDir;

    if ( sign * old_vel[c] > possmall ) {
      vel[c] = vel[cp];
    } else {
      vel[c] = zero;
    }
    vel[cm] = vel[c];
  }
}

template <class velType, class oldVelType> void
BoundaryCondition::outletPressureMinus( IntVector insideCellDir,
                                        Iterator bound_ptr,
                                        IntVector idxLo, IntVector idxHi,
                                        const int t1, const int t2,
                                        double const sign, velType& vel,
                                        oldVelType& old_vel,
                                        const bool mF1, const bool pF1,
                                        const bool mF2, const bool pF2 )
{

  double const negsmall = -1.0E-10;
  double const zero     = 0.0E0;

  for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

    IntVector c = *bound_ptr;
    IntVector cp  = c - insideCellDir;
    IntVector cpp = cp - insideCellDir;


    if ( (mF1 && (c[t1] == idxLo[t1])) ||
         (pF1 && (c[t1] == idxHi[t1])) ||
         (mF2 && (c[t2] == idxLo[t2])) ||
         (pF2 && (c[t2] == idxHi[t2])) ) {

      vel[cp] = zero;

    } else {

      if ( sign * old_vel[cp] < negsmall ) {
        vel[cp] = vel[cpp];
      } else {
        vel[cp] = zero;
      }
      vel[c] = vel[cp];
    }
  }
}
template <class velType, class oldVelType> void
BoundaryCondition::outletPressurePlus( IntVector insideCellDir,
                                       Iterator bound_ptr,
                                       IntVector idxLo, IntVector idxHi,
                                       const int t1, const int t2,
                                       double const sign, velType& vel,
                                       oldVelType& old_vel,
                                       const bool mF1, const bool pF1,
                                       const bool mF2, const bool pF2 )
{

  double possmall =  1.0E-10;
  double zero     = 0.0E0;
  for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

    IntVector c = *bound_ptr;
    IntVector cp  = c - insideCellDir;
    IntVector cm  = c + insideCellDir;

    if ( (mF1 && (c[t1] == idxLo[t1])) ||
         (pF1 && (c[t1] == idxHi[t1])) ||
         (mF2 && (c[t2] == idxLo[t2])) ||
         (pF2 && (c[t2] == idxHi[t2])) ) {

      vel[c] = zero;

    } else {

      if ( sign * old_vel[c] > possmall ) {
        vel[c] = vel[cp];
      } else {
        vel[c] = zero;
      }
      vel[cm] = vel[c];
    }
  }

}
template <class velType, class oldVelType> void
BoundaryCondition::neutralOutleMinus( IntVector insideCellDir,
                                      Iterator bound_ptr,
                                      velType& vel,
                                      oldVelType& old_vel )
{

  for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

    IntVector c = *bound_ptr;
    IntVector cp  = c - insideCellDir;
    IntVector cpp = cp - insideCellDir;

    vel[cp] = vel[cpp];
    vel[c] = vel[cp];

  }
}
template <class velType, class oldVelType> void
BoundaryCondition::neutralOutletPlus( IntVector insideCellDir,
                                      Iterator bound_ptr,
                                      velType& vel,
                                      oldVelType& old_vel )
{

  for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

    IntVector c = *bound_ptr;
    IntVector cp  = c - insideCellDir;
    IntVector cm  = c + insideCellDir;

    vel[c] = vel[cp];
    vel[cm] = vel[c];

  }
}

} // End namespace Uintah

#endif
