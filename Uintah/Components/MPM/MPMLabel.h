#ifndef UINTAH_HOMEBREW_MPMLABEL_H
#define UINTAH_HOMEBREW_MPMLABEL_H


#include <Uintah/Grid/VarLabel.h>

namespace Uintah {
  namespace MPM {
    class MPMLabel {
    public:

      MPMLabel();
      ~MPMLabel();

      static const MPMLabel* getLabels();

      const VarLabel* deltLabel;
      
      const VarLabel* pDeformationMeasureLabel;
      const VarLabel* pStressLabel;
      const VarLabel* pVolumeLabel;
      const VarLabel* pMassLabel;
      const VarLabel* pVelocityLabel;
      const VarLabel* pExternalForceLabel;
      const VarLabel* pXLabel;
      const VarLabel* pSurfLabel;
      const VarLabel* pSurfaceNormalLabel; //for fracture
      const VarLabel* pTemperatureLabel; //for heat conduction
      const VarLabel* pTemperatureGradientLabel; //for heat conduction
      
      const VarLabel* gMassLabel;
      const VarLabel* gAccelerationLabel;
      const VarLabel* gMomExedAccelerationLabel;
      const VarLabel* gVelocityLabel;
      const VarLabel* gMomExedVelocityLabel;
      const VarLabel* gVelocityStarLabel;
      const VarLabel* gMomExedVelocityStarLabel;
      const VarLabel* gExternalForceLabel;
      const VarLabel* gInternalForceLabel;
      const VarLabel* gSelfContactLabel; //for fracture
      const VarLabel* gTemperatureRateLabel; //for heat conduction
      const VarLabel* gInternalHeatRateLabel;
      const VarLabel* gExternalHeatRateLabel;
      
      const VarLabel* cSelfContactLabel; //for fracture, CCVariable
      const VarLabel* cSurfaceNormalLabel; //for fracture, CCVariable
      
      
    };
  } // end namepsace MPM
} // end namespace Uintah


// $Log$
// Revision 1.1  2000/05/26 21:37:30  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
#endif
