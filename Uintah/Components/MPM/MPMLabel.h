#ifndef UINTAH_HOMEBREW_MPMLABEL_H
#define UINTAH_HOMEBREW_MPMLABEL_H


#include <Uintah/Grid/VarLabel.h>
#include <vector>

using std::vector;

namespace Uintah {
  namespace MPM {
    class MPMLabel {
    public:

      MPMLabel();
      ~MPMLabel();

      void registerPermanentParticleState(int i,const VarLabel* l,
					  const VarLabel* lp);

      //      static const MPMLabel* getLabels();

      const VarLabel* delTLabel;
      
      const VarLabel* pDeformationMeasureLabel;
      const VarLabel* pStressLabel;
      const VarLabel* pVolumeLabel;
      const VarLabel* pVolumeDeformedLabel;
      const VarLabel* pMassLabel;
      const VarLabel* pVelocityLabel;
      const VarLabel* pExternalForceLabel;
      const VarLabel* pXLabel;
      const VarLabel* pSurfLabel;
      const VarLabel* pIsBrokenLabel; //for fracture
      const VarLabel* pCrackSurfaceNormalLabel; //for fracture
      const VarLabel* pMicrocrackSizeLabel; //for fracture
      const VarLabel* pTemperatureLabel; //for heat conduction
      const VarLabel* pTemperatureGradientLabel; //for heat conduction
      const VarLabel* pTemperatureRateLabel; //for heat conduction
      const VarLabel* pExternalHeatRateLabel; //for heat conduction
      const VarLabel* pParticleIDLabel;
      const VarLabel* pIsIgnitedLabel; //for burn models
      const VarLabel* pMassRateLabel; //for burn models

      const VarLabel* pDeformationMeasureLabel_preReloc;
      const VarLabel* pStressLabel_preReloc;
      const VarLabel* pVolumeLabel_preReloc;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pVelocityLabel_preReloc;
      const VarLabel* pExternalForceLabel_preReloc;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pSurfLabel_preReloc;
      const VarLabel* pIsBrokenLabel_preReloc; //for fracture
      const VarLabel* pCrackSurfaceNormalLabel_preReloc; //for fracture
      const VarLabel* pMicrocrackSizeLabel_preReloc; //for fracture
      const VarLabel* pTemperatureLabel_preReloc; //for heat conduction
      const VarLabel* pTemperatureGradientLabel_preReloc; //for heat conduction
      const VarLabel* pTemperatureRateLabel_preReloc; //for heat conduction
      const VarLabel* pExternalHeatRateLabel_preReloc; //for heat conduction
      const VarLabel* pParticleIDLabel_preReloc;
      const VarLabel* pIsIgnitedLabel_preReloc; //for burn models
      const VarLabel* pMassRateLabel_preReloc; //for burn models
      
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
      const VarLabel* gTemperatureLabel; //for heat conduction
      const VarLabel* gTemperatureStarLabel; //for heat conduction
      const VarLabel* gInternalHeatRateLabel;
      const VarLabel* gExternalHeatRateLabel;
      const VarLabel* gThermalContactHeatExchangeRateLabel;
      const VarLabel* gNormTractionLabel;
      const VarLabel* gSurfNormLabel;
      const VarLabel* gStressLabel;
      const VarLabel* gVolumeLabel; //for heat conduction
      const VarLabel* gWeightLabel; //for who knows what?
      
      const VarLabel* cBurnedMassLabel; //for burn models

      const VarLabel* StrainEnergyLabel;
      const VarLabel* KineticEnergyLabel;
      const VarLabel* TotalMassLabel;
      const VarLabel* CenterOfMassPositionLabel;
      const VarLabel* CenterOfMassVelocityLabel;

      const VarLabel* ppNAPIDLabel;

      vector<vector<const VarLabel* > > d_particleState;
      vector<vector<const VarLabel* > > d_particleState_preReloc;
    };
  } // end namepsace MPM
} // end namespace Uintah


// $Log$
// Revision 1.24  2000/09/07 21:11:04  tan
// Added particle variable pMicrocrackSize for fracture.
//
// Revision 1.23  2000/09/05 05:15:49  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.22  2000/08/30 00:12:42  guilkey
// Added some stuff for interpolating particle data to the grid solely
// for the purpose of saving to an uda.  This doesn't work yet.
//
// Revision 1.21  2000/08/08 19:55:59  tan
// Added cCrackedCellLabel and pCrackSurfaceNormalLabel for fracture.
//
// Revision 1.20  2000/08/04 16:42:42  guilkey
// Added VarLabels specific to FrictionContact so that those variables
// can be pleaseSaved.
//
// Revision 1.19  2000/07/27 22:17:16  jas
// Consolidated the registerPermanentParticleState to take both the
// regular labels and the pre_Reloc labels.
//
// Revision 1.18  2000/07/27 20:29:50  jas
// In SerialMPM.cc, problemSetup, there are now labels for each material.
// So it is now possible for different materials to have different VarLabels
// depending on the simulation requirements.
//
// Revision 1.17  2000/07/17 23:39:35  tan
// Fixed problems in MPM heat conduction.
//
// Revision 1.16  2000/07/12 18:45:07  jas
// Cleaned up the creation of the particle state and moved things into
// MPMLabel.
//
// Revision 1.15  2000/07/05 23:43:30  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.14  2000/06/28 01:08:56  tan
// Thermal contact model start to work!
//
// Revision 1.13  2000/06/27 21:50:57  guilkey
// Added saving of more "meta data."  Namely, center of mass position and
// velocity.  These are both actually mass weighted, so should be divided
// by total mass which is being saved in TotalMass.dat.
//
// Revision 1.12  2000/06/23 21:18:24  tan
// Added pExternalHeatRateLabel for heat conduction.
//
// Revision 1.11  2000/06/16 23:23:35  guilkey
// Got rid of pVolumeDeformedLabel_preReloc to fix some confusion
// the scheduler was having.
//
// Revision 1.10  2000/06/15 21:57:00  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.9  2000/06/13 23:06:38  guilkey
// Added a CCVariable for the burned particle mass to go into.
//
// Revision 1.8  2000/06/08 16:56:51  guilkey
// Added tasks and VarLabels for HE burn model stuff.
//
// Revision 1.7  2000/06/06 03:17:42  tan
// Added particle variable lable pAverageMicrocrackLength for fracture simulation.
//
// Revision 1.6  2000/06/02 23:16:32  guilkey
// Added ParticleID labels.
//
// Revision 1.5  2000/05/31 22:15:38  guilkey
// Added VarLabels for some integrated quantities.
//
// Revision 1.4  2000/05/31 16:11:11  tan
// gTemperatureLabel included
//
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
