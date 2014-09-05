#ifndef Fields_h
#define Fields_h

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <map>
#include <string>
#include <iostream>

#define YDIM
//#define ZDIM

//===========================================================================

namespace Uintah {
class VarLabel;
class Fields {
   
public:

  Fields();
 
  ~Fields();
  /** @brief Set the shared state */ 
  void setSharedState(SimulationStateP& sharedState);
  /** @brief Schedules a copy of old to new */
  void schedCopyOldToNew( const LevelP& level, SchedulerP& sched );
  /** @brief Allocates a new variable and copies old to new */ 
  void CopyOldToNew( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  SimulationStateP d_sharedState;

  struct PhysicalPropLabels {

    const VarLabel* lambda; 
    const VarLabel* density; 
    const VarLabel* temperature;

    // These may come in handy ..
    vector<const VarLabel*> myVec;
    std::map<std::string, const VarLabel*> myMap;

  };

  struct VelocityLabels {

    const VarLabel* uVelocity;
    const VarLabel* vVelocity; 
    const VarLabel* wVelocity; 

    const VarLabel* ccVelocity; 

    // These may come in handy ..
    vector<const VarLabel*> myVec;
    std::map<std::string, const VarLabel*> myMap;

  };

  PhysicalPropLabels propLabels;
  VelocityLabels     velocityLabels;  

  typedef map<string, const VarLabel* > LabelMap;
  LabelMap d_labelMap;

  //DQMOM velocity labels
  typedef map<int, const VarLabel* > PartVelMap;
  PartVelMap partVel;  

  /** @brief Interpolate a variable from it's storage point to the face of its
             respective volume.  XDIR */
  inline void interpToFace( CCVariable<double> ccVar, SFCXVariable<double> fcVar, Patch* patch )
  {
    //NOTE: this does not apply any boundary condition!
    //      just performs straight interpolation. 
    IntVector d(1,0,0); 

    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c  = *iter;
      
      fcVar[*iter] = 0.5 * ( ccVar[*iter] + ccVar[*iter - d] );
      //REPEATED WORK! --- is there a better way?
      fcVar[*iter + d] = 0.5 * ( ccVar[*iter] + ccVar[*iter + d] );
    }  
  }; 
   /** @brief Interpolate a variable from it's storage point to the face of its
             respective volume.  YDIR */
  inline void interpToFace( CCVariable<double> ccVar, SFCYVariable<double> fcVar, Patch* patch )
  {
    //NOTE: this does not apply any boundary condition!
    //      just performs straight interpolation. 
    IntVector d(0,1,0); 

    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c  = *iter;
      
      fcVar[*iter] = 0.5 * ( ccVar[*iter] + ccVar[*iter - d] );
      //REPEATED WORK! --- is there a better way?
      fcVar[*iter + d] = 0.5 * ( ccVar[*iter] + ccVar[*iter + d] );
    }  
  };
  /** @brief Interpolate a variable from it's storage point to the face of its
             respective volume.  ZDIR */
  inline void interpToFace( CCVariable<double> ccVar, SFCZVariable<double> fcVar, Patch* patch )
  {
    //NOTE: this does not apply any boundary condition!
    //      just performs straight interpolation. 
    IntVector d(0,0,1); 

    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c  = *iter;
      
      fcVar[*iter] = 0.5 * ( ccVar[*iter] + ccVar[*iter - d] );
      //REPEATED WORK! --- is there a better way?
      fcVar[*iter + d] = 0.5 * ( ccVar[*iter] + ccVar[*iter + d] );
    }  
  };
private:


 }; //end class Fields

} //end namespace Uintah
#endif 
