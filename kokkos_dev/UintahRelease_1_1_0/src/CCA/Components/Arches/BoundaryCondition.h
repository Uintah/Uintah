/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef Uintah_Components_Arches_BoundaryCondition_h
#define Uintah_Components_Arches_BoundaryCondition_h


#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>

#include   <vector>

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
   
   Copyright U of U 2000

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
class ExtraScalarSolver;

class BoundaryCondition {

public:
  
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a BoundaryCondition.
  // PRECONDITIONS
  // POSTCONDITIONS
  // Default constructor.
  BoundaryCondition();

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
  ////////////////////////////////////////////////////////////////////////
  int getNumSourceBndry() {
    return d_numSourceBoundaries;
  }

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

  bool getIntrusionBC() { 
    cout << "Intrusion machinery has been disabled" << endl;
    exit(1);
    return 1;
    //return d_intrusionBoundary; 
  }

  bool anyArchesPhysicalBC() { 
    return ((d_wallBoundary)||(d_inletBoundary)||(d_pressureBoundary)||(d_outletBoundary)||(d_intrusionBoundary)); 
  }

  ////////////////////////////////////////////////////////////////////////
  // Get the number of inlets (primary + secondary)
  int getNumInlets() { 
    return d_numInlets; 
  }

  ////////////////////////////////////////////////////////////////////////
  // mm Wall boundary ID
  int getMMWallId() const {
    return d_mmWallID;
  }

  ////////////////////////////////////////////////////////////////////////
  // flowfield cell id
  inline int flowCellType() const {
    return d_flowfieldCellTypeVal;
  }

  ////////////////////////////////////////////////////////////////////////
  // Wall boundary ID
  inline int wallCellType() const { 
    int wall_celltypeval = -10;
    if (d_wallBoundary){ 
      wall_celltypeval = d_wallBdry->d_cellTypeID; 
    }
    return wall_celltypeval;
  }

  ////////////////////////////////////////////////////////////////////////
  // Inlet boundary ID
  inline int inletCellType(int index) const {
    int inlet_celltypeval = -10;
    if ((d_inletBoundary)&&(index < d_numInlets)){
       inlet_celltypeval = d_flowInlets[index]->d_cellTypeID;
    }
    return inlet_celltypeval;
  }
  
  ////////////////////////////////////////////////////////////////////////
  // Pressure boundary ID
  inline int pressureCellType() const {
    int pressure_celltypeval = -10;
    if (d_pressureBoundary) pressure_celltypeval = d_pressureBC->d_cellTypeID; 
    return pressure_celltypeval;
  }

  ////////////////////////////////////////////////////////////////////////
  // Outlet boundary ID
  inline int outletCellType() const { 
    int outlet_celltypeval = -10;
    if (d_outletBoundary) outlet_celltypeval = d_outletBC->d_cellTypeID;
    return outlet_celltypeval; 
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

  inline double getIntrusionSourceVelocity(int whichIntrusion) {
    return d_sourceBoundaryInfo[whichIntrusion]->totalVelocity;
  }

  ////////////////////////////////////////////////////////////////////////
  // Access function for d_cutCells (multimaterial)
  inline bool getCutCells() const{
    return d_cutCells;
  }      

  inline bool getCarbonBalance() const{
    return d_carbon_balance;
  }
  
  inline bool getCarbonBalanceES() const{
    return d_carbon_balance_es;
  }
      
  inline bool getSulfurBalance() const{
    return d_sulfur_balance;
  } 
  inline bool getSulfurBalanceES() const{
    return d_sulfur_balance_es;
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

  ///////////////////////////////////////////////////////////////////////
  // Actually compute pressure BC terms
  void pressureBC(const Patch* patch,
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
  // applies multimaterial bc's for scalars and pressure
  void mmscalarWallBC__new( const Patch* patch,
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
                        double time_shift);

  void velRhoHatOutletPressureBC(const Patch* patch,
                                 ArchesVariables* vars,
                                 ArchesConstVariables* constvars);

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
  inline void setCalcExtraScalars(bool calcExtraScalars) {
    d_calcExtraScalars=calcExtraScalars;
  }
  inline void setExtraScalars(vector<ExtraScalarSolver*>* extraScalars) {
    d_extraScalars = extraScalars;
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

private:

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

private:
  // GROUP:  Local DataTypes :
  ////////////////////////////////////////////////////////////////////////
  // FlowInlet
  class FlowInlet {
  public:
    FlowInlet();
    FlowInlet(const FlowInlet& copy);
    FlowInlet(int cellID, bool calcVariance, bool reactingScalarSolve);
    ~FlowInlet();
    FlowInlet& operator=(const FlowInlet& copy);
    int d_cellTypeID;          // define enum for cell type
    bool d_calcVariance;
    bool d_reactingScalarSolve;
    // inputs
    double flowRate;           
    double inletVel;           
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

  ////////////////////////////////////////////////////////////////////////
  // Intrusion Boundary
  struct IntrusionBdry {
    int d_cellTypeID;
    double area;
    double d_temperature;
    // stores the geometry information, read from problem specs
    std::vector<GeometryPieceP> d_geomPiece;
    IntrusionBdry(int cellID);
    IntrusionBdry() {}
    void problemSetup(ProblemSpecP& params);
  };

  //*-------------------------------------*
  // BCSourceInfo
  // a struct to hold infromation for a specific
  // geometry piece that applies a source term 
  // on the surface of itself
  //*-------------------------------------*
  class BCSourceInfo
  {
    public:
    BCSourceInfo();
    BCSourceInfo(bool calcVariance, bool reactingScalarSolve);
    ~BCSourceInfo();
              
    //The geometry piece          
    std::vector<GeometryPieceP> d_geomPiece;
    //Area information
    double area_x; //total area with normals in the x-direction
    double area_y; //total area with normals in the y-direction
    double area_z; //total area with normals in the z-direction
    VarLabel* total_area_label; //total area of all directions
    double summed_area;
    bool computedArea; //a bool to tell the code if the area has 
                       // been computed for this particular object 

    //Normal information
    Vector normal;
    //Flux information
    double umom_flux; //velocities
    double vmom_flux;
    double wmom_flux;
    double f_flux;   //mixture fraction
    double h_flux;   //enthalpy
    double totalMassFlux;
    double totalVelocity;
    double totalFlowArea;
    string velocityType;
    string velocityRelation;
    InletStream streamMixturefraction; //inlet values
    Stream calcStream; // calculated values
    bool d_calcVariance;
    bool d_reactingScalarSolve;

    //Mixture fraction inlet value
    double mixfrac_inlet;

    //relational information
    Vector axisStart;
    Vector axisEnd;
    Vector point;
    bool doAreaCalc;

    //---methods---                
    //Problem setup
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

  bool d_intrusionBoundary;
  IntrusionBdry* d_intrusionBC;

  bool d_carbon_balance;    //Use table value of CO2
  bool d_carbon_balance_es; //Extra scalar for carbon balance
  bool d_sulfur_balance;
  bool d_sulfur_balance_es; //Extra scalar for sulfur balance
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
  bool d_calcExtraScalars;
  vector<ExtraScalarSolver*>* d_extraScalars;

  struct d_extraScalarBC {
    string d_scalar_name;
    double d_scalarBC_value;
    int d_BC_ID; 
  };
  vector<d_extraScalarBC*> d_extraScalarBCs; 

  //BC source term stuff
  std::vector<BCSourceInfo* > d_sourceBoundaryInfo;
  int d_numSourceBoundaries;

  typedef std::map<std::string, struct EfficiencyInfo> EfficiencyMap;
  EfficiencyMap d_effVars;

  typedef std::map<std::string, struct SpeciesEfficiencyInfo> SpeciesEffMap; // label string, molecular weight ratio 
  SpeciesEffMap d_speciesEffInfo;

}; // End of class BoundaryCondition
} // End namespace Uintah

#endif  
  
