
#ifndef Uintah_Components_Arches_Source_h
#define Uintah_Components_Arches_Source_h

/**************************************
CLASS
   Source
   
   Class Source computes source terms for 
   N-S equations.  

GENERAL INFORMATION
   Source.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class Source computes source terms for 
   N-S equations.  

WARNING
   none

****************************************/

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>

#include <Core/Containers/Array1.h>
namespace Uintah {
  class ProcessorGroup;
class PhysicalConstants;
class TurbulenceModel;
using namespace SCIRun;

class Source {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a Source.
      // PRECONDITIONS
      // POSTCONDITIONS
      Source();

      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a Source.
      // PRECONDITIONS
      // POSTCONDITIONS
      Source(TurbulenceModel* turb_model, PhysicalConstants* phys_const);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~Source();

      // GROUP:  Action Methods
      ////////////////////////////////////////////////////////////////////////
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      void calculatePressureSource(const ProcessorGroup* pc,
				   const Patch* patch,
				   double delta_t,
				   CellInformation* cellinfo,
				   ArchesVariables* vars); 

      void calculatePressureSourcePred(const ProcessorGroup* pc,
				       const Patch* patch,
				       double delta_t,
				       CellInformation* cellinfo,
				       ArchesVariables* vars,
				       ArchesConstVariables* constvars); 

      void calculatePressureSourceCorr(const ProcessorGroup* pc,
				   const Patch* patch,
				   double delta_t,
				   CellInformation* cellinfo,
				   ArchesVariables* vars); 

      void calculateVelocityPred(const ProcessorGroup* ,
				 const Patch* patch,
				 double delta_t,
				 int index,
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      void calculateVelocitySource(const ProcessorGroup* pc,
				   const Patch* patch,
				   double delta_t, 
				   int index,
				   CellInformation* cellinfo,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      void calculateScalarSource(const ProcessorGroup* pc,
				 const Patch* patch,
				 double delta_t, 
				 int index,
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

      void addReactiveScalarSource(const ProcessorGroup*,
				   const Patch* patch,
				   double delta_t,
				   int, 
				   CellInformation* cellinfo,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars);


      void calculateEnthalpySource(const ProcessorGroup* pc,
				 const Patch* patch,
				 double delta_t, 
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

      void computeEnthalpyRadFluxes(const ProcessorGroup* pc,
				    const Patch* patch,
				    CellInformation* cellinfo,
				    ArchesVariables* vars);

      void computeEnthalpyRadSrc(const ProcessorGroup* pc,
				 const Patch* patch,
				 CellInformation* cellinfo,
				 ArchesVariables* vars);

      void computeEnthalpyRadThinSrc(const ProcessorGroup* pc,
				     const Patch* patch,
				     CellInformation* cellinfo,
				     ArchesVariables* vars,
				     ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      void modifyVelMassSource(const ProcessorGroup* pc,
			       const Patch* patch,
			       double delta_t, 
			       int index,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      void modifyScalarMassSource(const ProcessorGroup* pc,
				  const Patch* patch,
				  double delta_t, 
				  int index, ArchesVariables* vars,
				  ArchesConstVariables* constvars,
				  int conv_scheme);

      void modifyEnthalpyMassSource(const ProcessorGroup* pc,
				  const Patch* patch,
				  double delta_t, 
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars,
				  int conv_scheme);

      ////////////////////////////////////////////////////////////////////////
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      void addPressureSource(const ProcessorGroup* pc,
			     const Patch* patch,
			     double delta_t,
			     int index,
			     CellInformation* cellinfo,
			     ArchesVariables* vars);
      // add transient term in momentum source term
      void addTransMomSource(const ProcessorGroup* ,
			     const Patch* patch ,
			     double delta_t,
			     int index,
			     CellInformation* cellinfo,			  
			     ArchesVariables* vars);

      void computePressureSource(const ProcessorGroup* ,
				 const Patch* patch ,
				 int index,
				 CellInformation* cellinfo,			  
				 ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Add multimaterial source term
      void computemmMomentumSource(const ProcessorGroup* pc,
				   const Patch* patch,
				   int index,
				   CellInformation* cellinfo,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars);
  
      void addMMEnthalpySource(const ProcessorGroup* pc,
			    const Patch* patch,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars);
  
private:

      TurbulenceModel* d_turbModel;
      PhysicalConstants* d_physicalConsts;


}; // end Class Source

} // End namespace Uintah
#endif  
  
