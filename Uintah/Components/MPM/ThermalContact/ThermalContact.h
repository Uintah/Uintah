#ifndef __ThermalContact__
#define __ThermalContact__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <math.h>

namespace Uintah {
   using SCICore::Geometry::Vector;
   using SCICore::Math::Min;

   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;
   namespace MPM {
     class MPMMaterial;

/**************************************

CLASS
   ThermalContact
   
   Short description...

GENERAL INFORMATION

   ThermalContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ThermalContact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

class ThermalContact {
public:
  // Constructor
  ThermalContact(ProblemSpecP& ps,SimulationStateP& d_sS);
  ~ThermalContact();

  void computeHeatExchange(const ProcessorGroup*,
                           const Patch* patch,
                           DataWarehouseP& old_dw,
                           DataWarehouseP& new_dw);
	 
  void initializeThermalContact(const Patch* patch,
				int vfindex,
				DataWarehouseP& new_dw);

  void addComputesAndRequires(Task* task,
                              const MPMMaterial* matl,
                              const Patch* patch,
                              DataWarehouseP& old_dw,
                              DataWarehouseP& new_dw) const;

private:
  SimulationStateP d_sharedState;
  MPMLabel* lb;
};
      
} // end namespace MPM
} // end namespace Uintah

// $Log$
// Revision 1.7  2000/08/09 03:18:03  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.6  2000/07/05 23:43:38  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.5  2000/06/20 05:09:44  tan
// Currently thermal_conductivity, specific_heat and heat_transfer_coefficient
// are set in MPM::MPMMaterial class.
//
// Revision 1.4  2000/06/20 03:40:51  tan
// Get thermal_conductivity, specific_heat and heat_transfer_coefficient
// from ProblemSpecification input requires.
//
// Revision 1.3  2000/06/17 07:06:41  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.2  2000/05/31 22:29:23  tan
// Finished addComputesAndRequires function.
//
// Revision 1.1  2000/05/31 18:16:39  tan
// Create ThermalContact class to handle heat exchange in
// contact mechanics.
//

#endif // __ThermalContact__

