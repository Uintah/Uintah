
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

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
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

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>

#include <SCIRun/Core/Containers/Array1.h>
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

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      void problemSetup(const ProblemSpecP& params);

      // GROUP:  Action Methods
      ////////////////////////////////////////////////////////////////////////
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      void calculatePressureSourcePred(const ProcessorGroup* pc,
				       const Patch* patch,
				       double delta_t,
				       CellInformation* cellinfo,
				       ArchesVariables* vars,
				       ArchesConstVariables* constvars,
                                       bool doing_EKT_now); 
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
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

      void addReactiveScalarSource(const ProcessorGroup*,
				   const Patch* patch,
				   double delta_t,
				   CellInformation* cellinfo,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars);

      void calculateEnthalpySource(const ProcessorGroup* pc,
				 const Patch* patch,
				 double delta_t, 
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

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
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars,
				  int conv_scheme);

      void modifyEnthalpyMassSource(const ProcessorGroup* pc,
				  const Patch* patch,
				  double delta_t, 
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars,
				  int conv_scheme);


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
  
      void calculateVelMMSSource(const ProcessorGroup* pc,
				   const Patch* patch,
				   double delta_t, double time,
				   int index,
				   CellInformation* cellinfo,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars);

      void calculateScalarMMSSource(const ProcessorGroup* pc,
				 const Patch* patch,
				 double delta_t, 
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

      void calculatePressMMSSourcePred(const ProcessorGroup* pc,
				       const Patch* patch,
				       double delta_t,
				       CellInformation* cellinfo,
				       ArchesVariables* vars,
				       ArchesConstVariables* constvars); 

private:

      TurbulenceModel* d_turbModel;
      PhysicalConstants* d_physicalConsts;
      string d_mms;
      double d_airDensity, d_heDensity;
      Vector d_gravity;
      double d_viscosity;
      double d_turbPrNo;

      // linear mms
      double cu, cv, cw, cp, phi0;
      // sine mms
      double amp;


}; // end Class Source

} // End namespace Uintah
#endif  
  
