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
#ifndef included_BoundaryCondition
#define included_BoundaryCondition
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
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

class BoundaryCondition
{
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
   
   BoundaryCondition();

   BoundaryCondition(TurbulenceModel* turb_model, Properties* props);
  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~BoundaryCondition();
   
   void problemSetup(const ProblemSpecP& params);
   // initialize celltyping
   void BoundaryCondition::cellTypeInit(const ProcessorContext*,
				   const Patch* patch,
				   DataWarehouseP& old_dw,  DataWarehouseP&);
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Set boundary conditions terms. 
   
   void sched_velocityBC(const int index,
			 const LevelP& level,
			 SchedulerP& sched,
			 const DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);
   void sched_pressureBC(const LevelP& level,
			 const Patch* patch,
			 const DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);
   void sched_scalarBC(const int index,
		       const LevelP& level,
		       SchedulerP& sched,
		       const DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw);
   // Set inlet velocity bc's, we need to do it because of staggered grid
   // 
   void sched_setInletVelocityBC(const LevelP& level,
				 SchedulerP& sched,
				 const DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw);
   // used for pressure boundary type
   void sched_computePressureBC(const LevelP& level,
				SchedulerP& sched,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);
   // initializes velocities, scalars and properties at the bndry
   // remove sched_setFlat and sched_setProps
   void sched_setProfile(const LevelP& level,
			 SchedulerP& sched,
			 const DataWarehouseP& old_dw);

   // assign flat velocity profiles at the inlet
   void sched_setFlatProfile(const LevelP& level,
			     SchedulerP& sched,
			     const DataWarehouseP& old_dw);
   void sched_setPropsProfile(const LevelP& level,
			     SchedulerP& sched,
			     const DataWarehouseP& old_dw);
 private:
   void setFlatProfile(const ProcessorContext* pc,
		       const Patch* patch,
		       const DataWarehouseP& old_dw);
   void setPropsProfile(const ProcessorContext* pc,
		       const Patch* patch,
		       const DataWarehouseP& old_dw);
   void setInletVelocityBC(const ProcessorContext* pc,
			   const Patch* patch,
			   const DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);
   void calculatePressBC(const ProcessorContext* pc,
			 const Patch* patch,
			 const DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);
   void velocityBC(const ProcessorContext* pc,
		   const Patch* patch,
		   const DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw,
		   const int index);
   void pressureBC(const ProcessorContext*,
		   const Patch* patch,
		   const DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw);
   void scalarBC(const ProcessorContext* pc,
		 const Patch* patch,
		 const DataWarehouseP& old_dw,
		 DataWarehouseP& new_dw,
		 const int index);
   // used for calculating wall boundary conditions
   TurbulenceModel* d_turbModel;
   // used to get properties of different streams
   Properties* d_props;
   struct FlowInlet {
     // define enum for cell type
     int d_cellTypeID;
     // input vars
     double flowRate;
     // array of size numMixingVars -1
     std::vector<double> streamMixturefraction;
     double turb_lengthScale;
     // calculated values
     double density;
     // inlet area
     double area;
     // stores the geometry information, read from problem specs
     GeometryPiece* d_geomPiece;
     FlowInlet(int numMix, int cellID);
     void problemSetup(ProblemSpecP& params);
   };
   struct PressureInlet {
     int d_cellTypeID;
     // array of size numMixingVars -1
     std::vector<double> streamMixturefraction;
     double turb_lengthScale;
     double density;
     double refPressure;
     double area;
     // stores the geometry information, read from problem specs
     GeometryPiece* d_geomPiece;
     PressureInlet(int numMix, int cellID);
     void problemSetup(ProblemSpecP& params);
   };
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
   struct WallBdry {
     int d_cellTypeID;
     double area;
     // stores the geometry information, read from problem specs
     GeometryPiece* d_geomPiece;
     WallBdry(int cellID);
     void problemSetup(ProblemSpecP& params);
   };

   // variable labels
   const VarLabel* d_cellTypeLabel;
   std::vector<int> d_cellTypes;
   WallBdry* d_wallBdry;
   int d_numInlets;
   int d_numMixingScalars;
   std::vector<FlowInlet> d_flowInlets;
   bool d_pressBoundary;
   PressureInlet* d_pressureBdry;
   bool d_outletBoundary;
   FlowOutlet* d_outletBC;
};
}
}
#endif  
  
