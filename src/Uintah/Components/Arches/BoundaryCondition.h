//
// $Id$
//

#ifndef Uintah_Components_Arches_BoundaryCondition_h
#define Uintah_Components_Arches_BoundaryCondition_h

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

#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Grid/Array3.h>
#include <SCICore/Containers/Array1.h>
#include <vector>

namespace Uintah {

class VarLabel;
namespace MPM {
  class GeometryPiece;
}

namespace ArchesSpace {

using namespace SCICore::Containers;
using namespace Uintah::MPM;
class TurbulenceModel;
class Properties;

class BoundaryCondition {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of a BoundaryCondition.
      //
      // PRECONDITIONS
      //
      //
      // POSTCONDITIONS
      //
      // Default constructor.
      //
      BoundaryCondition();

      ////////////////////////////////////////////////////////////////////////
      //
      // BoundaryCondition constructor used in Uintah PSE
      //
      BoundaryCondition(TurbulenceModel* turb_model, Properties* props);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      ~BoundaryCondition();
   
      // GROUP: Problem Steup:
      ////////////////////////////////////////////////////////////////////////
      //
      // Details here
      //
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Access function
      ////////////////////////////////////////////////////////////////////////
      //
      // Details here
      //
      int getNumInlets() {
	return d_numInlets;
      }

      // GROUP: Access function
      ////////////////////////////////////////////////////////////////////////
      //
      // Details here
      //
      int getFlowCellID(int index) {
	return d_flowInlets[index].d_cellTypeID;
      }

      ////////////////////////////////////////////////////////////////////////
      //
      // Initialize celltyping
      // Details here
      //
      void cellTypeInit(const ProcessorContext*,
			const Patch* patch,
			DataWarehouseP& old_dw,  
			DataWarehouseP&);
      ////////////////////////////////////////////////////////////////////////
      //
      // computing inlet areas
      // Details here
      //
      void computeInletFlowArea(const ProcessorContext*,
			const Patch* patch,
			DataWarehouseP& old_dw,  
			DataWarehouseP&);

      // GROUP:  Schedule tasks :
      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule Computation of Velocity boundary conditions terms. 
      //
      void sched_velocityBC(const LevelP& level,
			    SchedulerP& sched,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw,
			    int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule Computation of Pressure boundary conditions terms. 
      //
      void sched_pressureBC(const LevelP& level,
			    SchedulerP& sched,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule Computation of Scalar boundary conditions terms. 
      //
      void sched_scalarBC(const LevelP& level,
			  SchedulerP& sched,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw,
			  int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule Setting inlet velocity bc's
      // we need to do it because of staggered grid
      // 
      void sched_setInletVelocityBC(const LevelP& level,
				    SchedulerP& sched,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule Compute Pressure BCS
      // used for pressure boundary type
      //
      void sched_computePressureBC(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule Set Profile BCS
      // initializes velocities, scalars and properties at the bndry
      // assigns flat velocity profiles for primary and secondary inlets
      // Also sets flat profiles for density
      // ** WARNING ** Properties profile not done yet
      //
      void sched_setProfile(const LevelP& level,
			    SchedulerP& sched,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

private:

      // GROUP:  Actual Computations (Private)  :
      ////////////////////////////////////////////////////////////////////////
      //
      // Actually compute velocity BC terms
      //
      void velocityBC(const ProcessorContext* pc,
		      const Patch* patch,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw,
		      const int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Call Fortran to compute u velocity BC terms
      //
      void uVelocityBC(DataWarehouseP& new_dw,
		       const Patch* patch,
		       const IntVector& indexLow, 
		       const IntVector& indexHigh,
		       CCVariable<int>* cellType,
		       CCVariable<double>* uVelocity, 
		       CCVariable<double>* vVelocity, 
		       CCVariable<double>* wVelocity, 
		       CCVariable<double>* density,
		       const double* VISCOS,
		       CellInformation* cellinfo);

      ////////////////////////////////////////////////////////////////////////
      //
      // Call Fortran to compute v velocity BC terms
      //
      void vVelocityBC(DataWarehouseP& new_dw,
		       const Patch* patch,
		       const IntVector& indexLow, 
		       const IntVector& indexHigh,
		       CCVariable<int>* cellType,
		       CCVariable<double>* uVelocity, 
		       CCVariable<double>* vVelocity, 
		       CCVariable<double>* wVelocity, 
		       CCVariable<double>* density,
		       const double* VISCOS,
		       CellInformation* cellinfo);

      ////////////////////////////////////////////////////////////////////////
      //
      // Call Fortran to compute w velocity BC terms
      //
      void wVelocityBC(DataWarehouseP& new_dw,
		       const Patch* patch,
		       const IntVector& indexLow, 
		       const IntVector& indexHigh,
		       CCVariable<int>* cellType,
		       CCVariable<double>* uVelocity, 
		       CCVariable<double>* vVelocity, 
		       CCVariable<double>* wVelocity, 
		       CCVariable<double>* density,
		       const double* VISCOS,
		       CellInformation* cellinfo);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually compute pressure BC terms
      //
      void pressureBC(const ProcessorContext*,
		      const Patch* patch,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually compute scalar BC terms
      //
      void scalarBC(const ProcessorContext* pc,
		    const Patch* patch,
		    DataWarehouseP& old_dw,
		    DataWarehouseP& new_dw,
		    const int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually set inlet velocity bcs
      //
      void setInletVelocityBC(const ProcessorContext* pc,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually calculate pressure bcs
      //
      void calculatePressBC(const ProcessorContext* pc,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually set the velocity, density and props flat profile
      //
      void setFlatProfile(const ProcessorContext* pc,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw);

private:

      // GROUP:  Local DataTypes :
      ////////////////////////////////////////////////////////////////////////
      //
      // FlowInlet
      //
      struct FlowInlet {
	int d_cellTypeID;          // define enum for cell type
	// inputs
	double flowRate;           
        std::vector<double> streamMixturefraction; // array [numMixingVars-1]
	double turb_lengthScale;
	// calculated values
	double density;
	// stores the geometry information, read from problem specs
	GeometryPiece* d_geomPiece;
	FlowInlet(int numMix, int cellID);
	void problemSetup(ProblemSpecP& params);
	// reduction variable label to get area
	const VarLabel* d_area_label;
      };

      ////////////////////////////////////////////////////////////////////////
      //
      // PressureInlet
      //
      struct PressureInlet {
	int d_cellTypeID;
	std::vector<double> streamMixturefraction; // array [numMixingVars-1]
	double turb_lengthScale;
	double density;
	double refPressure;
	double area;
	// stores the geometry information, read from problem specs
	GeometryPiece* d_geomPiece;
	PressureInlet(int numMix, int cellID);
	void problemSetup(ProblemSpecP& params);
      };

      ////////////////////////////////////////////////////////////////////////
      //
      // FlowOutlet
      //
      struct FlowOutlet {
	int d_cellTypeID;
	std::vector<double> streamMixturefraction;
	double turb_lengthScale;
	double density;
	double area;
	// stores the geometry information, read from problem specs
	GeometryPiece* d_geomPiece;
	FlowOutlet(int numMix, int cellID);
	void problemSetup(ProblemSpecP& params);
      };

      ////////////////////////////////////////////////////////////////////////
      //
      // Wall Boundary
      //
      struct WallBdry {
	int d_cellTypeID;
	double area;
	// stores the geometry information, read from problem specs
	GeometryPiece* d_geomPiece;
	WallBdry(int cellID);
	void problemSetup(ProblemSpecP& params);
      };

private:

      // used for calculating wall boundary conditions
      TurbulenceModel* d_turbModel;
      // used to get properties of different streams
      Properties* d_props;

      // variable labels
      std::vector<int> d_cellTypes;
      WallBdry* d_wallBdry;
      int d_numInlets;
      int d_numMixingScalars;
      std::vector<FlowInlet> d_flowInlets;
      bool d_pressBoundary;
      PressureInlet* d_pressureBdry;
      bool d_outletBoundary;
      FlowOutlet* d_outletBC;

      // const VarLabel*
      const VarLabel* d_cellTypeLabel;
      const VarLabel* d_pressureLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_uVelocityLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_vVelocityLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_wVelocityLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_uVelCoefLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_vVelCoefLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_wVelCoefLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_uVelLinSrcLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_vVelLinSrcLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_wVelLinSrcLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_uVelNonLinSrcLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_vVelNonLinSrcLabel;
      //** WARNING ** Velocity is a FC Variable
      const VarLabel* d_wVelNonLinSrcLabel;
      const VarLabel* d_presCoefLabel;

}; // End of class BoundaryCondition
} // End namespace ArchesSpace
} // End namespace Uintah
#endif  
  
//
// $Log$
// Revision 1.21  2000/06/15 22:13:22  rawat
// modified boundary stuff
//
// Revision 1.20  2000/06/15 08:48:12  bbanerje
// Removed most commented stuff , added StencilMatrix, tasks etc.  May need some
// more work
//
//
