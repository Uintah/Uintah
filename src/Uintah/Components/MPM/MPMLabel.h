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

      const VarLabel* delTLabel;
      
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
      const VarLabel* pTemperatureRateLabel; //for heat conduction
      
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
// Revision 1.3  2000/05/30 17:07:34  dav
// Removed commented out labels.  Other MPI fixes.  Changed delt to delT so I would stop thinking of it as just delta.
//
// Revision 1.2  2000/05/30 04:27:33  tan
// Added gTemperatureRateLabel for heat conduction computations.
//
// Revision 1.1  2000/05/26 21:37:30  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
#endif
