#ifndef __FRACTURE_H__
#define __FRACTURE_H__

#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>

#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {

   class VarLabel;
   class ProcessorContext;
   class Region;

namespace MPM {

class Fracture {
public:
  void   materialDefectsInitialize();
  
  void   updateSurfaceNormalOfBoundaryParticle(
           const ProcessorContext*,
           const Region* region,
           const DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);
  
  void   crackGrow(
           const ProcessorContext*,
           const Region* region,
           const DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);

         Fracture();
	 Fracture(ProblemSpecP& ps, SimulationStateP& d_sS);
                
private:
  double           d_averageMicrocrackLength;
  double           d_materialToughness;

  VarLabel*        pSurfaceNormalLabel; 
  VarLabel*        pStressLabel; 
  VarLabel*        pDeformationMeasureLabel;
  VarLabel*        pXLabel; 
  VarLabel*        cSelfContactLabel;

  SimulationStateP d_sharedState;
};

} //namespace MPM
} //namespace Uintah

#endif //__FRACTURE_H__

// $Log$
// Revision 1.4  2000/05/10 05:04:39  tan
// Basic structure of fracture class.
//
