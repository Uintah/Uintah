//----- Properties.h --------------------------------------------------

#ifndef Uintah_Component_Arches_Properties_h
#define Uintah_Component_Arches_Properties_h

/***************************************************************************
CLASS
    Properties
       Sets up the Properties ????
       
GENERAL INFORMATION
    Properties.h - Declaration of Properties class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>

#include <vector>

namespace Uintah {
namespace ArchesSpace {

class Properties {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      Properties();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      ~Properties();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Compute properties 
      ///////////////////////////////////////////////////////////////////////
      //
      // Compute properties for inlet/outlet streams
      //
      double computeInletProperties(const std::vector<double>&
				    mixfractionStream);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the computation of proprties
      //
      void sched_computeProps(const LevelP& level,
			      SchedulerP&, 
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw);

      ///////////////////////////////////////////////////////////////////////
      //
      // Schedule the recomputation of proprties
      //
      void sched_reComputeProps(const LevelP& level,
				SchedulerP&, 
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      int getNumMixVars() const{ return d_numMixingVars; }

protected :

private:

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      void computeProps(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw);

      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual recomputation of properties
      //
      void reComputeProps(const ProcessorGroup*,
			const Patch* patch,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw);

      // GROUP: Constructors Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const Properties&   
      //
      Properties(const Properties&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const Properties&   
      //
      Properties& operator=(const Properties&);

private:

      int d_numMixingVars;
      double d_denUnderrelax;

      struct Stream {
	double d_density;
	double d_temperature;
	Stream();
	void problemSetup(ProblemSpecP&);
      };
      std::vector<Stream> d_streams; 

      // Variable labels used by simulation controller
      const VarLabel* d_densitySPLabel;   // Input density
      const VarLabel* d_densityCPLabel;   // Output density
      const VarLabel* d_densitySIVBCLabel;   // Input density
      const VarLabel* d_densityRCPLabel;   // Output density
      const VarLabel* d_scalarSPLabel;
}; // end class Properties

} // end namespace ArchesSpace
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.13  2000/06/30 04:19:17  rawat
// added turbulence model and compute properties
//
// Revision 1.12  2000/06/19 18:00:30  rawat
// added function to compute velocity and density profiles and inlet bc.
// Fixed bugs in CellInformation.cc
//
// Revision 1.11  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.10  2000/06/17 07:06:25  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.9  2000/06/16 21:50:48  bbanerje
// Changed the Varlabels so that sequence in understood in init stage.
// First cycle detected in task graph.
//
// Revision 1.8  2000/05/31 08:12:45  bbanerje
// Added Cocoon stuff to Properties, added VarLabels, changed task, requires,
// computes, get etc.in Properties, changed fixed size Mixing Var array to
// vector.
//
//
