
#ifndef Uintah_Components_Arches_BoundaryCondition_h
#define Uintah_Components_Arches_BoundaryCondition_h

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Containers/Array1.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

/**************************************
CLASS
   BoundaryCondition
   
   Class BoundaryCondition applies boundary conditions
   at physical boundaries. For boundary cell types it
   modifies stencil coefficients and source terms.

GENERAL INFORMATION
   BoundaryCondition.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
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
class GeometryPiece;
class PhysicalConstants;
class Properties;
class Stream;
class InletStream;
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
  class DataWarehouse;
  class TimeIntegratorLabel;

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
      BoundaryCondition(const ArchesLabel* label, const MPMArchesLabel* MAlb,
			PhysicalConstants* phyConsts, Properties* props,
			bool calcReactScalar, bool calcEnthalpy);

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
      bool getWallBC() { return d_wallBoundary; }
      
      bool getInletBC() { return d_inletBoundary; }
      
      bool getPressureBC() { return d_pressureBoundary; }

      bool getOutletBC() { return d_outletBoundary; }
      
      bool getIntrusionBC() { return d_intrusionBoundary; }

      bool anyArchesPhysicalBC() { 
       return ((d_wallBoundary)||(d_inletBoundary)||(d_pressureBoundary)||(d_outletBoundary)||(d_intrusionBoundary)); }
      
      ////////////////////////////////////////////////////////////////////////
      // Get the number of inlets (primary + secondary)
      int getNumInlets() { return d_numInlets; }


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
	if (d_wallBoundary) wall_celltypeval = d_wallBdry->d_cellTypeID; 
	return wall_celltypeval;
      }

      ////////////////////////////////////////////////////////////////////////
      // Inlet boundary ID
      inline int inletCellType(int index) const {
	int inlet_celltypeval = -10;
	if ((d_inletBoundary)&&(index < d_numInlets))
	   inlet_celltypeval = d_flowInlets[index].d_cellTypeID;
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
      void setIfCalcEnergyExchange(bool calcEnergyExchange)
	{
	  d_calcEnergyExchange = calcEnergyExchange;
	}

      ////////////////////////////////////////////////////////////////////////
      // Access function for calcEnergyExchange (multimaterial)

      inline bool getIfCalcEnergyExchange() const{
	return d_calcEnergyExchange;
      }      

      // GROUP:  Schedule tasks :
      ////////////////////////////////////////////////////////////////////////
      // Initialize cell types
      void sched_cellTypeInit(SchedulerP&, const PatchSet* patches,
			      const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Initialize inlet area
      // Details here
      void sched_calculateArea(SchedulerP&, const PatchSet* patches,
			       const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Schedule Computation of Pressure boundary conditions terms. 
      void sched_computePressureBC(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Schedule Set Profile BCS
      // initializes velocities, scalars and properties at the bndry
      // assigns flat velocity profiles for primary and secondary inlets
      // Also sets flat profiles for density
      // ** WARNING ** Properties profile not done yet
      void sched_setProfile(SchedulerP&, const PatchSet* patches,
			    const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multimaterial wall cell types
      void sched_mmWallCellTypeInit( SchedulerP&, const PatchSet* patches,
				     const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multimaterial wall cell types for first time step
      void sched_mmWallCellTypeInit_first( SchedulerP&, const PatchSet* patches,
					   const MaterialSet* matls);

      // GROUP:  Actual Computations :
      ////////////////////////////////////////////////////////////////////////
      // Initialize celltyping
      // Details here
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

      ////////////////////////////////////////////////////////////////////////
      // Actually compute velocity BC terms
      void velocityBC(const ProcessorGroup* pc,
		      const Patch* patch,
		      int index,
		      CellInformation* cellinfo,
		      ArchesVariables* vars,
		      ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute pressure BC terms
      void pressureBC(const ProcessorGroup*,
		      const Patch* patch,
		      DataWarehouse* old_dw,
		      DataWarehouse* new_dw,
		      CellInformation* cellinfo,
		      ArchesVariables* vars,
		      ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute scalar BC terms
      void scalarBC(const ProcessorGroup* pc,
		    const Patch* patch,
		    int index,
		    CellInformation* cellinfo,
		    ArchesVariables* vars,
		    ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute enthalpy BC terms
      void enthalpyBC(const ProcessorGroup* pc,
		      const Patch* patch,
		      CellInformation* cellinfo,
		      ArchesVariables* vars,
		      ArchesConstVariables* constvars);

      void enthalpyRadWallBC(const ProcessorGroup* pc,
			     const Patch* patch,
			     CellInformation* cellinfo,
			     ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multi-material wall celltyping and void fraction 
      // calculation
      void mmWallCellTypeInit(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multi-material wall celltyping and void fraction 
      // calculation for first time step
      void mmWallCellTypeInit_first(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw);
      // for computing intrusion bc's

      void intrusionTemperatureBC(const ProcessorGroup*,
				  const Patch* patch,
				  constCCVariable<int>& cellType,
				  CCVariable<double>& temperature);

      void mmWallTemperatureBC(const ProcessorGroup*,
			       const Patch* patch,
			       constCCVariable<int>& cellType,
			       constCCVariable<double> solidTemp,
			       CCVariable<double>& temperature);

      void calculateIntrusionVel(const ProcessorGroup*,
				 const Patch* patch,
				 int index,
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
		      	         ArchesConstVariables* constvars);

      void intrusionVelocityBC(const ProcessorGroup*,
			       const Patch* patch,
			       int index, CellInformation* cellinfo,
			       ArchesVariables* vars,
		      	       ArchesConstVariables* constvars);

      void intrusionMomExchangeBC(const ProcessorGroup*,
				  const Patch* patch,
				  int index, CellInformation* cellinfo,
				  ArchesVariables* vars,
		      		  ArchesConstVariables* constvars);

      void intrusionEnergyExBC(const ProcessorGroup*,
			       const Patch* patch,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
		      	       ArchesConstVariables* constvars);

      void intrusionPressureBC(const ProcessorGroup*,
			       const Patch* patch,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars);

      void intrusionScalarBC(const ProcessorGroup*,
			     const Patch* patch,
			     CellInformation* cellinfo,
			     ArchesVariables* vars,
			     ArchesConstVariables* constvars);

      void intrusionEnthalpyBC(const ProcessorGroup*,
			       const Patch* patch, double delta_t,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars);

      // compute multimaterial wall bc
      void mmvelocityBC(const ProcessorGroup*,
			const Patch* patch,
			int index, CellInformation* cellinfo,
			ArchesVariables* vars,
		       	ArchesConstVariables* constvars);

      void mmpressureBC(const ProcessorGroup*,
			const Patch* patch,
			CellInformation* cellinfo,
			ArchesVariables* vars,
		       	ArchesConstVariables* constvars);

      // applies multimaterial bc's for scalars and pressure
      void mmscalarWallBC( const ProcessorGroup*,
			   const Patch* patch,
			   CellInformation* cellinfo,
			   ArchesVariables* vars,
			   ArchesConstVariables* constvars);
      
      ////////////////////////////////////////////////////////////////////////
      // Calculate uhat for multimaterial case (only for nonintrusion cells)
      void calculateVelRhoHat_mm(const ProcessorGroup* /*pc*/,
				 const Patch* patch,
				 int index, double delta_t,
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

      // adds pressure gradient to momentum nonlinear source term
      void addPressureGrad(const ProcessorGroup* ,
			   const Patch* patch ,
			   int index,
			   CellInformation* cellinfo,			  
			   ArchesVariables* vars);


      void calculateVelocityPred_mm(const ProcessorGroup*,
				    const Patch* patch,
				    double delta_t,
				    int index,
				    CellInformation* cellinfo,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars);

      void setFluxBC(const ProcessorGroup* pc,
		      const Patch* patch,
		      int index,
		      ArchesVariables* vars);
			
      void scalarLisolve_mm(const ProcessorGroup*,
			    const Patch*,
			    double delta_t,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    CellInformation* cellinfo);
			
      void enthalpyLisolve_mm(const ProcessorGroup*,
			      const Patch*,
			      double delta_t,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars,
			      CellInformation* cellinfo);
// New boundary conditions
      void scalarPressureBC(const ProcessorGroup* pc,
		    const Patch* patch,
		    int index,
		    CellInformation* cellinfo,
		    ArchesVariables* vars,
		    ArchesConstVariables* constvars,
			    const double delta_t);

      void enthalpyPressureBC(const ProcessorGroup* pc,
		    const Patch* patch,
		    CellInformation* cellinfo,
		    ArchesVariables* vars,
		    ArchesConstVariables* constvars);

      void scalarOutletBC(const ProcessorGroup* pc,
			    const Patch* patch,
			    int index,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    const double delta_t,
			    const double maxAbsU,
			    const double maxAbsV,
			    const double maxAbsW);

      void enthalpyOutletBC(const ProcessorGroup* pc,
			    const Patch* patch,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    const double delta_t,
			    const double maxAbsU,
			    const double maxAbsV,
			    const double maxAbsW);

      void velRhoHatInletBC(const ProcessorGroup* pc,
			    const Patch* patch,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars);

      void velRhoHatPressureBC(const ProcessorGroup* pc,
			    const Patch* patch,
			    CellInformation* cellinfo,
			    const double delta_t,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars);

      void velRhoHatOutletBC(const ProcessorGroup* pc,
			    const Patch* patch,
			    CellInformation* cellinfo,
			    const double delta_t,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    const double maxAbsU,
			    const double maxAbsV,
			    const double maxAbsW);

      void velocityPressureBC(const ProcessorGroup* pc,
			    const Patch* patch,
			    const int index,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars);

      void addPresGradVelocityOutletBC(const ProcessorGroup* pc,
			    const Patch* patch,
			    const int index,
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
private:

      // GROUP:  Actual Computations (Private)  :
      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute u velocity BC terms
      void uVelocityBC(const Patch* patch,
		       double VISCOS,
		       CellInformation* cellinfo,
		       ArchesVariables* vars,
		       ArchesConstVariables* constvars);
		      

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute v velocity BC terms
      void vVelocityBC(const Patch* patch,
		       double VISCOS,
		       CellInformation* cellinfo,
		       ArchesVariables* vars,
		       ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute w velocity BC terms
      void wVelocityBC(const Patch* patch,
		       double VISCOS,
		       CellInformation* cellinfo,
		       ArchesVariables* vars,
		       ArchesConstVariables* constvars);

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
      // Call Fortran to compute u velocity BC terms
      void mmuVelocityBC(const Patch* patch,
			 ArchesVariables* vars,
		       	 ArchesConstVariables* constvars);
		      

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute v velocity BC terms
      void mmvVelocityBC(const Patch* patch,
			 ArchesVariables* vars,
		       	 ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute w velocity BC terms
      void mmwVelocityBC(const Patch* patch,
			 ArchesVariables* vars,
		       	 ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Actually calculate pressure bcs
      void computePressureBC(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);


      ////////////////////////////////////////////////////////////////////////
      // Actually set the velocity, density and props flat profile
      void setFlatProfile(const ProcessorGroup* pc,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);

// New boundary conditions
      void getFlowINOUT(const ProcessorGroup* pc,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw,
			const TimeIntegratorLabel* timelabels);

      void correctVelocityOutletBC(const ProcessorGroup* pc,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw,
			      const TimeIntegratorLabel* timelabels);
private:

      // GROUP:  Local DataTypes :
      ////////////////////////////////////////////////////////////////////////
      // FlowInlet
      class FlowInlet {
      public:
	FlowInlet();
	FlowInlet(const FlowInlet& copy);
	FlowInlet(int numMix, int cellID);
	~FlowInlet();
	FlowInlet& operator=(const FlowInlet& copy);
	int d_cellTypeID;          // define enum for cell type
	// inputs
	double flowRate;           
	double inletVel;           
        InletStream streamMixturefraction; // array [numMixingVars-1]
	double turb_lengthScale;
	// calculated values
	Stream calcStream;
	// stores the geometry information, read from problem specs
	std::vector<GeometryPiece*> d_geomPiece;
	void problemSetup(ProblemSpecP& params);
	// reduction variable label to get area
	VarLabel* d_area_label;
      };

      ////////////////////////////////////////////////////////////////////////
      // PressureInlet
      struct PressureInlet {
	int d_cellTypeID;
	InletStream streamMixturefraction; // array [numMixingVars-1]
	double turb_lengthScale;
	Stream calcStream;
	double refPressure;
	double area;
	// stores the geometry information, read from problem specs
	std::vector<GeometryPiece*> d_geomPiece;
	PressureInlet(int numMix, int cellID);
	void problemSetup(ProblemSpecP& params);
      };

      ////////////////////////////////////////////////////////////////////////
      // FlowOutlet
      struct FlowOutlet {
	int d_cellTypeID;
	InletStream streamMixturefraction; // array [numMixingVars-1]
	double turb_lengthScale;
	Stream calcStream;
	double area;
	// stores the geometry information, read from problem specs
	std::vector<GeometryPiece*> d_geomPiece;
	FlowOutlet(int numMix, int cellID);
	void problemSetup(ProblemSpecP& params);
      };

      ////////////////////////////////////////////////////////////////////////
      // Wall Boundary
      struct WallBdry {
	int d_cellTypeID;
	double area;
	// stores the geometry information, read from problem specs
	std::vector<GeometryPiece*> d_geomPiece;
	WallBdry(int cellID);
	void problemSetup(ProblemSpecP& params);
      };

      ////////////////////////////////////////////////////////////////////////
      // Intrusion Boundary
      struct IntrusionBdry {
	int d_cellTypeID;
	double area;
	double d_temperature;
	// stores the geometry information, read from problem specs
	std::vector<GeometryPiece*> d_geomPiece;
	IntrusionBdry(int cellID);
	void problemSetup(ProblemSpecP& params);
      };

private:

      // const VarLabel* inputs
      const ArchesLabel* d_lab;
      // for multimaterial
      const MPMArchesLabel* d_MAlab;
      int d_mmWallID;
      // cutoff for void fraction rqd to determine multimaterial wall
      double MM_CUTOFF_VOID_FRAC;
      bool d_calcEnergyExchange;

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
      // variable labels
      std::vector<int> d_cellTypes;
      int d_flowfieldCellTypeVal;

      bool d_wallBoundary;
      WallBdry* d_wallBdry;
      
      bool d_inletBoundary;
      int d_numInlets;
      int d_numMixingScalars;
      int d_nofScalars;
      std::vector<FlowInlet> d_flowInlets;

      bool d_pressureBoundary;
      PressureInlet* d_pressureBC;

      bool d_outletBoundary;
      FlowOutlet* d_outletBC;

      bool d_intrusionBoundary;
      IntrusionBdry* d_intrusionBC;

}; // End of class BoundaryCondition
} // End namespace Uintah

#endif  
  
