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

#include <Uintah/Components/Arches/ArchesLabel.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Array3.h>
#include <SCICore/Containers/Array1.h>
#include <Uintah/Components/Arches/ArchesVariables.h>
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
      BoundaryCondition(const ArchesLabel* label,
			TurbulenceModel* turb_model, Properties* props);

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

      // GROUP: Access functions
      ////////////////////////////////////////////////////////////////////////
      //
      // Returns pressureBC flag
      //
      int getPressureBC() { return d_pressBoundary; }

      ////////////////////////////////////////////////////////////////////////
      //
      // Get the number of inlets (primary + secondary)
      //
      int getNumInlets() { return d_numInlets; }

      ////////////////////////////////////////////////////////////////////////
      //
      // Details here
      //
      int getFlowCellID(int index) {
	return d_flowInlets[index].d_cellTypeID;
      }

      // GROUP:  Schedule tasks :
      ////////////////////////////////////////////////////////////////////////
      //
      // Initialize cell types
      //
      void sched_cellTypeInit(const LevelP& level,
			      SchedulerP& sched,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Initialize inlet area
      // Details here
      //
      void sched_calculateArea(const LevelP& level,
			       SchedulerP& sched,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Schedule Computation of Pressure boundary conditions terms. 
      //
      void sched_computePressureBC(const LevelP& level,
			    SchedulerP& sched,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

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
      // used for pressure boundary type (during time advance)
      //
      void sched_recomputePressureBC(const LevelP& level,
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

      // GROUP:  Actual Computations :
      ////////////////////////////////////////////////////////////////////////
      //
      // Initialize celltyping
      // Details here
      //
      void cellTypeInit(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,  
			DataWarehouseP&);
      ////////////////////////////////////////////////////////////////////////
      //
      // computing inlet areas
      // Details here
      //
      void computeInletFlowArea(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,  
			DataWarehouseP&);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually compute velocity BC terms
      //
      void velocityBC(const ProcessorGroup* pc,
		      const Patch* patch,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw,
		      const int index,
		      int eqnType,
		      CellInformation* cellinfo,
		      ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually compute pressure BC terms
      //
      void pressureBC(const ProcessorGroup*,
		      const Patch* patch,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw,
		      CellInformation* cellinfo,
		      ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually compute scalar BC terms
      //
      void scalarBC(const ProcessorGroup* pc,
		    const Patch* patch,
		    DataWarehouseP& old_dw,
		    DataWarehouseP& new_dw,
		    const int index,
		    CellInformation* cellinfo,
		    ArchesVariables* vars);

private:

      // GROUP:  Actual Computations (Private)  :
      ////////////////////////////////////////////////////////////////////////
      //
      // Call Fortran to compute u velocity BC terms
      //
      void uVelocityBC(DataWarehouseP& new_dw,
		       const Patch* patch,
		       const double* VISCOS,
		       CellInformation* cellinfo,
		       ArchesVariables* vars);
		      

      ////////////////////////////////////////////////////////////////////////
      //
      // Call Fortran to compute v velocity BC terms
      //
      void vVelocityBC(DataWarehouseP& new_dw,
		       const Patch* patch,
		       const double* VISCOS,
		       CellInformation* cellinfo,
		       ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Call Fortran to compute w velocity BC terms
      //
      void wVelocityBC(DataWarehouseP& new_dw,
		       const Patch* patch,
		       const double* VISCOS,
		       CellInformation* cellinfo,
		        ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually set inlet velocity bcs
      //
      void setInletVelocityBC(const ProcessorGroup* pc,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually calculate pressure bcs
      //
      void calcPressureBC(const ProcessorGroup* pc,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually calculate pressure bcs
      //
      void recomputePressureBC(const ProcessorGroup* pc,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Actually set the velocity, density and props flat profile
      //
      void setFlatProfile(const ProcessorGroup* pc,
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
	std::vector<GeometryPiece*> d_geomPiece;
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
	std::vector<GeometryPiece*> d_geomPiece;
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
	std::vector<GeometryPiece*> d_geomPiece;
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
	std::vector<GeometryPiece*> d_geomPiece;
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
      int d_nofScalars;
      std::vector<FlowInlet> d_flowInlets;
      bool d_pressBoundary;
      PressureInlet* d_pressureBdry;
      bool d_outletBoundary;
      FlowOutlet* d_outletBC;
      int d_flowfieldCellTypeVal;

      // const VarLabel* inputs
      const ArchesLabel* d_lab;

}; // End of class BoundaryCondition
} // End namespace ArchesSpace
} // End namespace Uintah
#endif  
  
//
// $Log$
// Revision 1.40  2000/07/28 02:30:59  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.37  2000/07/17 22:06:58  rawat
// modified momentum source
//
// Revision 1.36  2000/07/14 03:45:45  rawat
// completed velocity bc and fixed some bugs
//
// Revision 1.35  2000/07/12 23:59:21  rawat
// added wall bc for u-velocity
//
// Revision 1.34  2000/07/08 08:03:33  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.33  2000/07/07 23:07:45  rawat
// added inlet bc's
//
// Revision 1.32  2000/07/03 05:30:14  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.31  2000/07/02 05:47:30  bbanerje
// Uncommented all PerPatch and CellInformation stuff.
// Updated array sizes in inlbcs.F
//
// Revision 1.30  2000/06/30 06:29:42  bbanerje
// Got Inlet Area to be calculated correctly .. but now two CellInformation
// variables are being created (Rawat ... check that).
//
// Revision 1.29  2000/06/29 06:22:48  bbanerje
// Updated FCVariable to SFCX, SFCY, SFCZVariables and made corresponding
// changes to profv.  Code is broken until the changes are reflected
// thru all the files.
//
// Revision 1.28  2000/06/28 08:14:53  bbanerje
// Changed the init routines a bit.
//
// Revision 1.27  2000/06/22 23:06:33  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.26  2000/06/19 18:00:29  rawat
// added function to compute velocity and density profiles and inlet bc.
// Fixed bugs in CellInformation.cc
//
// Revision 1.25  2000/06/18 01:20:14  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.24  2000/06/17 07:06:23  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.23  2000/06/16 21:50:47  bbanerje
// Changed the Varlabels so that sequence in understood in init stage.
// First cycle detected in task graph.
//
// Revision 1.22  2000/06/16 04:25:39  bbanerje
// Uncommented BoundaryCondition related stuff.
//
// Revision 1.21  2000/06/15 22:13:22  rawat
// modified boundary stuff
//
// Revision 1.20  2000/06/15 08:48:12  bbanerje
// Removed most commented stuff , added StencilMatrix, tasks etc.  May need some
// more work
//
//
