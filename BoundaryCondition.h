
#ifndef Uintah_Components_Arches_BoundaryCondition_h
#define Uintah_Components_Arches_BoundaryCondition_h

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Containers/Array1.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <vector>

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
  class CellInformation;
class VarLabel;
class GeometryPiece;
class TurbulenceModel;
class Properties;
class Stream;
class InletStream;
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
  class DataWarehouse;

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
			TurbulenceModel* turb_model, Properties* props,
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
      // Returns pressureBC flag
      int getPressureBC() { return d_pressBoundary; }

      ////////////////////////////////////////////////////////////////////////
      // Get the number of inlets (primary + secondary)
      int getNumInlets() { return d_numInlets; }

      ////////////////////////////////////////////////////////////////////////
      // Details here
      int getFlowCellID(int index) {
	return d_flowInlets[index].d_cellTypeID;
      }

      ////////////////////////////////////////////////////////////////////////
      // mm Wall boundary ID
      int getMMWallId() const {
	return d_mmWallID;
      }

      ////////////////////////////////////////////////////////////////////////
      // flowfield cell id
      int getFlowId() const {
	return d_flowfieldCellTypeVal;
      }

        ////////////////////////////////////////////////////////////////////////
      // Wall boundary ID
      inline int wallCellType() const { 
	return d_wallBdry->d_cellTypeID; 
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
      // Schedule Setting inlet velocity bc's
      // we need to do it because of staggered grid
      void sched_setInletVelocityBC(SchedulerP&, const PatchSet* patches,
				    const MaterialSet* matls);

      void sched_computeFlowINOUT(SchedulerP& sched, const PatchSet* patches,
				  const MaterialSet* matls);
      void sched_computeOMB(SchedulerP& sched, const PatchSet* patches,
			    const MaterialSet* matls);

      void sched_transOutletBC(SchedulerP& sched, 
			       const PatchSet* patches,
			       const MaterialSet* matls);

      ////////////////////////////////////////////////////////////////////////
      // Schedule Compute Pressure BCS
      // used for pressure boundary type (during time advance)
      void sched_recomputePressureBC(SchedulerP&, const PatchSet* patches,
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
		      const int index,
		      CellInformation* cellinfo,
		      ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute pressure BC terms
      void pressureBC(const ProcessorGroup*,
		      const Patch* patch,
		      DataWarehouse* old_dw,
		      DataWarehouse* new_dw,
		      CellInformation* cellinfo,
		      ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute scalar BC terms
      void scalarBC(const ProcessorGroup* pc,
		    const Patch* patch,
		    const int index,
		    CellInformation* cellinfo,
		    ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Actually compute enthalpy BC terms
      void enthalpyBC(const ProcessorGroup* pc,
		      const Patch* patch,
		      CellInformation* cellinfo,
		      ArchesVariables* vars);

      void enthalpyRadWallBC(const ProcessorGroup* pc,
			     const Patch* patch,
			     CellInformation* cellinfo,
			     ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Initialize multi-material wall celltyping
      // Details here
      void mmWallCellTypeInit(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);


      // compute multimaterial wall bc
      void mmvelocityBC(const ProcessorGroup*,
			const Patch* patch,
			int index, CellInformation* cellinfo,
			ArchesVariables* vars);

      void mmpressureBC(const ProcessorGroup*,
			const Patch* patch,
			CellInformation* cellinfo,
			ArchesVariables* vars);
      // applies multimaterial bc's for scalars and pressure
      void mmscalarWallBC( const ProcessorGroup*,
			   const Patch* patch,
			   CellInformation* cellinfo,
			   ArchesVariables* vars);
      
      // adds pressure gradient to momentume nonlinear source term
      void addPressureGrad(const ProcessorGroup* ,
			   const Patch* patch ,
			   int index,
			   CellInformation* cellinfo,			  
			   ArchesVariables* vars);

      void newrecomputePressureBC(const ProcessorGroup* /*pc*/,
				  const Patch* patch,
				  CellInformation* cellinfo,
				  ArchesVariables* vars);
			
private:

      // GROUP:  Actual Computations (Private)  :
      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute u velocity BC terms
      void uVelocityBC(const Patch* patch,
		       double VISCOS,
		       CellInformation* cellinfo,
		       ArchesVariables* vars);
		      

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute v velocity BC terms
      void vVelocityBC(const Patch* patch,
		       double VISCOS,
		       CellInformation* cellinfo,
		       ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute w velocity BC terms
      void wVelocityBC(const Patch* patch,
		       double VISCOS,
		       CellInformation* cellinfo,
		        ArchesVariables* vars);

      void mmuVelocityBC(const Patch* patch,
			 ArchesVariables* vars);
		      

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute v velocity BC terms
      void mmvVelocityBC(const Patch* patch,
			 ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Call Fortran to compute w velocity BC terms
      void mmwVelocityBC(const Patch* patch,
			 ArchesVariables* vars);
      ////////////////////////////////////////////////////////////////////////
      // Actually set inlet velocity bcs
      void setInletVelocityBC(const ProcessorGroup* pc,
			       const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

      void transOutletBC(const ProcessorGroup* ,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw,
			 DataWarehouse* new_dw);

      void computeFlowINOUT(const ProcessorGroup* pc,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);
  
      void computeOMB(const ProcessorGroup* pc,
		      const PatchSubset* patches,
		      const MaterialSubset* matls,
		      DataWarehouse* old_dw,
		      DataWarehouse* new_dw);

      ////////////////////////////////////////////////////////////////////////
      // Actually calculate pressure bcs
      void calcPressureBC(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);

      ////////////////////////////////////////////////////////////////////////
      // Actually calculate pressure bcs
      void recomputePressureBC(const ProcessorGroup* pc,
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

private:

      // GROUP:  Local DataTypes :
      ////////////////////////////////////////////////////////////////////////
      // FlowInlet
      struct FlowInlet {
	int d_cellTypeID;          // define enum for cell type
	// inputs
	double flowRate;           
        InletStream streamMixturefraction; // array [numMixingVars-1]
	double turb_lengthScale;
	// calculated values
	Stream calcStream;
	// stores the geometry information, read from problem specs
	std::vector<GeometryPiece*> d_geomPiece;
	FlowInlet(int numMix, int cellID);
	void problemSetup(ProblemSpecP& params);
	// reduction variable label to get area
	const VarLabel* d_area_label;
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

private:

      // used for calculating wall boundary conditions
      TurbulenceModel* d_turbModel;
      // used to get properties of different streams
      Properties* d_props;
      // mass flow
      double d_uvwout;
      double d_overallMB;
      // for enthalpy solve 
      bool d_enthalpySolve;
      // for reacting scalar
      bool d_reactingScalarSolve;
      // variable labels
      std::vector<int> d_cellTypes;
      WallBdry* d_wallBdry;
      int d_numInlets;
      int d_numMixingScalars;
      int d_nofScalars;
      std::vector<FlowInlet> d_flowInlets;
      bool d_pressBoundary;
      PressureInlet* d_pressureBdry;
      bool d_outletBoundary;
      FlowOutlet* d_outletBC;
      int d_flowfieldCellTypeVal;

      // const VarLabel* inputs
      const ArchesLabel* d_lab;
      // for multimaterial
      const MPMArchesLabel* d_MAlab;
      int d_mmWallID;
      // cutoff for void fraction rqd to determine multimaterial wall
      double MM_CUTOFF_VOID_FRAC;

}; // End of class BoundaryCondition
} // End namespace Uintah

#endif  
  
