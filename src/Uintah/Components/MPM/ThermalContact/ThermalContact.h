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

#include <math.h>

namespace Uintah {
   using SCICore::Geometry::Vector;
   using SCICore::Math::Min;

   class ProcessorContext;
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
  ThermalContact();

  void computeHeatExchange(const ProcessorContext*,
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
};
      
} // end namespace MPM
} // end namespace Uintah
   
// $Log$
// Revision 1.2  2000/05/31 22:29:23  tan
// Finished addComputesAndRequires function.
//
// Revision 1.1  2000/05/31 18:16:39  tan
// Create ThermalContact class to handle heat exchange in
// contact mechanics.
//

#endif // __ThermalContact__

