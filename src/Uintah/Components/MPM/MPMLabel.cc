#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/PerPatch.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace Uintah::MPM;

MPMLabel::MPMLabel()
{
  // Particle Variables

  //non PermanentParticleState
  pDeformationMeasureLabel = scinew VarLabel("p.deformationMeasure",
			ParticleVariable<Matrix3>::getTypeDescription());

  pRotationRateLabel = scinew VarLabel("p.rotationRate",
			ParticleVariable<Vector>::getTypeDescription());

  pVisibilityLabel = scinew VarLabel("p.visibility",
			ParticleVariable<int>::getTypeDescription());
  
  pStressReleasedLabel = scinew VarLabel("p.stressReleased",
			ParticleVariable<int>::getTypeDescription());
  
  pIsNewlyBrokenLabel = scinew VarLabel("p.isNewlyBroken",
			ParticleVariable<int>::getTypeDescription());

  pStressAfterStrainRateLabel = scinew VarLabel("p.stressAfterStrainRate",
			ParticleVariable<Matrix3>::getTypeDescription());

  pStressAfterFractureReleaseLabel = scinew VarLabel("p.stressAfterFractureRelease",
			ParticleVariable<Matrix3>::getTypeDescription());

  pVelocityAfterUpdateLabel = scinew VarLabel("p.velocityAfterUpdate",
			ParticleVariable<Vector>::getTypeDescription());

  pVelocityAfterFractureLabel = scinew VarLabel("p.velocityAfterFracture",
			ParticleVariable<Vector>::getTypeDescription());

  pStrainEnergyLabel = scinew VarLabel("p.strainEnergy",
			ParticleVariable<double>::getTypeDescription());

  pNewlyBrokenSurfaceNormalLabel = scinew VarLabel("p.newlyBrokenSurfaceNormal",
			ParticleVariable<Vector>::getTypeDescription());

  pXXLabel = scinew VarLabel("p.positionXX",
			ParticleVariable<Point>::getTypeDescription());
  
  //PermanentParticleState
  pStressLabel = scinew VarLabel( "p.stress",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pVolumeLabel = scinew VarLabel( "p.volume",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeDeformedLabel = scinew VarLabel( "p.volumedeformed",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel = scinew VarLabel( "p.mass",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel = scinew VarLabel( "p.velocity", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel = scinew VarLabel( "p.externalforce",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel = scinew VarLabel("p.x",ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);
  
  pTemperatureLabel = scinew VarLabel( "p.temperature",
			ParticleVariable<double>::getTypeDescription() );
  
  pTemperatureGradientLabel = scinew VarLabel( "p.temperatureGradient",
			ParticleVariable<Vector>::getTypeDescription() );

  pTemperatureRateLabel  = scinew VarLabel( "p.temperatureRate",
			ParticleVariable<double>::getTypeDescription() );

  pExternalHeatRateLabel = scinew VarLabel( "p.externalHeatRate",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel = scinew VarLabel( "p.surface",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel = scinew VarLabel( "p.isBroken",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel = scinew VarLabel( "p.crackNormal",
			ParticleVariable<Vector>::getTypeDescription() );

  pCrackSurfaceContactForceLabel = scinew VarLabel("p.crackSurfaceContactForce",
			ParticleVariable<Vector>::getTypeDescription());

  pTensileStrengthLabel = scinew VarLabel( "p.tensileStrength",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel = scinew VarLabel( "p.energyReleaseRateLabel",
			ParticleVariable<double>::getTypeDescription() );

  pImageVelocityLabel = scinew VarLabel( "p.imageVelocity",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel = scinew VarLabel("p.particleID",
			ParticleVariable<long>::getTypeDescription() );

  pIsIgnitedLabel  = scinew VarLabel( "p.isIgnited",
			ParticleVariable<int>::getTypeDescription() );
  
  pMassRateLabel  = scinew VarLabel( "p.massRate",
			ParticleVariable<double>::getTypeDescription() );
  
  // Particle Variables 
  pDeformationMeasureLabel_preReloc = scinew VarLabel("p.deformationMeasure+",
			ParticleVariable<Matrix3>::getTypeDescription());
  
  pStressLabel_preReloc = scinew VarLabel( "p.stress+",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pVolumeLabel_preReloc = scinew VarLabel( "p.volume+",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel_preReloc = scinew VarLabel( "p.mass+",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel_preReloc = scinew VarLabel( "p.velocity+", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel_preReloc = scinew VarLabel( "p.externalforce+",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel_preReloc = scinew VarLabel( "p.x+", ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);
  
  pTemperatureLabel_preReloc = scinew VarLabel( "p.temperature+",
			ParticleVariable<double>::getTypeDescription() );
  
  pTemperatureGradientLabel_preReloc = scinew VarLabel( "p.temperatureGradient+",
			ParticleVariable<Vector>::getTypeDescription() );

  pTemperatureRateLabel_preReloc  = scinew VarLabel( "p.temperatureRate+",
			ParticleVariable<double>::getTypeDescription() );

  pExternalHeatRateLabel_preReloc = scinew VarLabel( "p.externalHeatRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel_preReloc = scinew VarLabel( "p.surface+",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel_preReloc = scinew VarLabel( "p.isBroken+",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel_preReloc = scinew VarLabel( "p.crackNormal+",
			ParticleVariable<Vector>::getTypeDescription() );

  pCrackSurfaceContactForceLabel_preReloc = scinew VarLabel("p.crackSurfaceContactForce+",
			ParticleVariable<Vector>::getTypeDescription());

  pTensileStrengthLabel_preReloc = scinew VarLabel( "p.tensileStrength+",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel_preReloc = scinew VarLabel( "p.energyReleaseRateLabel+",
			ParticleVariable<double>::getTypeDescription() );

  pImageVelocityLabel_preReloc = scinew VarLabel( "p.imageVelocity+",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel_preReloc = scinew VarLabel("p.particleID+",
			ParticleVariable<long>::getTypeDescription() );

  pIsIgnitedLabel_preReloc  = scinew VarLabel( "p.isIgnited+",
			ParticleVariable<int>::getTypeDescription() );
  
  pMassRateLabel_preReloc  = scinew VarLabel( "p.massRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  // Node Centered Variables
  
  gAccelerationLabel = scinew VarLabel( "g.acceleration",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedAccelerationLabel = scinew VarLabel( "g.momexedacceleration",
			NCVariable<Vector>::getTypeDescription() );
  
  gMassLabel = scinew VarLabel( "g.mass",
			NCVariable<double>::getTypeDescription() );
  
  gVelocityLabel = scinew VarLabel( "g.velocity",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityLabel = scinew VarLabel( "g.momexedvelocity",
			NCVariable<Vector>::getTypeDescription() );
  
  gExternalForceLabel = scinew VarLabel( "g.externalforce",
			NCVariable<Vector>::getTypeDescription() );
  
  gInternalForceLabel = scinew VarLabel( "g.internalforce",
			NCVariable<Vector>::getTypeDescription() );
  
  gVelocityStarLabel = scinew VarLabel( "g.velocity_star",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityStarLabel = scinew VarLabel( "g.momexedvelocity_star",
			NCVariable<Vector>::getTypeDescription() );
  
  gSelfContactLabel = scinew VarLabel( "g.selfContact",
			NCVariable<bool>::getTypeDescription() );

  gCrackNormalLabel = scinew VarLabel( "g.crackNormal",
			NCVariable<Vector>::getTypeDescription() );

  gTensileStrengthLabel = scinew VarLabel( "g.tensileStrength",
			NCVariable<double>::getTypeDescription() );
  
  gTemperatureLabel = scinew VarLabel("g.temperature",
			NCVariable<double>::getTypeDescription());

  gTemperatureStarLabel = scinew VarLabel("g.temperatureStar",
			NCVariable<double>::getTypeDescription());

  gTemperatureRateLabel = scinew VarLabel("g.temperatureRate",
			NCVariable<double>::getTypeDescription());

  gInternalHeatRateLabel = scinew VarLabel("g.internalHeatRate",
			NCVariable<double>::getTypeDescription());

  gExternalHeatRateLabel = scinew VarLabel("g.externalHeatRate",
			NCVariable<double>::getTypeDescription());

  gThermalContactHeatExchangeRateLabel = scinew 
     VarLabel("g.thermalContactHeatExchangeRate",
     NCVariable<double>::getTypeDescription());

  gNormTractionLabel = scinew VarLabel( "g.normtraction",
                   NCVariable<double>::getTypeDescription() );

  gSurfNormLabel = scinew VarLabel( "g.surfnorm",
                   NCVariable<Vector>::getTypeDescription() );

  gStressLabel   = scinew VarLabel( "g.stress",
                   NCVariable<Matrix3>::getTypeDescription() );

  gStressForSavingLabel   = scinew VarLabel( "g.stressFS",
                   NCVariable<Matrix3>::getTypeDescription() );

  gVolumeLabel = scinew VarLabel("g.volume",
			NCVariable<double>::getTypeDescription());

  gWeightLabel = scinew VarLabel("g.weight",
			NCVariable<double>::getTypeDescription());

  // Cell centered variables
  cBurnedMassLabel = scinew VarLabel( "c.burnedMass",
			CCVariable<double>::getTypeDescription() );

  // Reduction variables

  mom_L_ME_CCLabel =
     scinew VarLabel("mom_L_ME_CC",CCVariable<Vector>::getTypeDescription());

  // Reduction variables
  delTLabel = scinew VarLabel( "delT", delt_vartype::getTypeDescription() );

  StrainEnergyLabel = scinew VarLabel( "StrainEnergy",
			sum_vartype::getTypeDescription() );

  KineticEnergyLabel = scinew VarLabel( "KineticEnergy",
			sum_vartype::getTypeDescription() );

  TotalMassLabel = scinew VarLabel( "TotalMass",
				 sum_vartype::getTypeDescription() );

  NTractionZMinusLabel = scinew VarLabel( "NTractionZMinus",
			sum_vartype::getTypeDescription() );

  CenterOfMassPositionLabel = scinew VarLabel( "CenterOfMassPosition",
				 sumvec_vartype::getTypeDescription() );

  CenterOfMassVelocityLabel = scinew VarLabel( "CenterOfMassVelocity",
				 sumvec_vartype::getTypeDescription() );

  // PerPatch variables

  ppNAPIDLabel = scinew VarLabel("NAPID",PerPatch<long>::getTypeDescription() );

} 

MPMLabel::~MPMLabel()
{
  //non PermanentParticleState
  delete pDeformationMeasureLabel;
  delete pRotationRateLabel,
  delete pVisibilityLabel;
  delete pStressReleasedLabel;
  delete pIsNewlyBrokenLabel;
  
  delete pStressAfterStrainRateLabel;
  delete pStressAfterFractureReleaseLabel;

  delete pVelocityAfterUpdateLabel;
  delete pVelocityAfterFractureLabel;
  
  delete pStrainEnergyLabel;
  delete pNewlyBrokenSurfaceNormalLabel;
  
  delete pXXLabel;

  //PermanentParticleState
  delete pStressLabel;
  delete pVolumeLabel;
  delete pVolumeDeformedLabel;
  delete pMassLabel;
  delete pVelocityLabel;
  delete pExternalForceLabel;
  delete pXLabel;
  delete pTemperatureLabel;
  delete pTemperatureGradientLabel;
  delete pTemperatureRateLabel;
  delete pExternalHeatRateLabel;
  delete pSurfLabel;
  delete pIsBrokenLabel;
  delete pCrackNormalLabel;
  delete pCrackSurfaceContactForceLabel;
  delete pTensileStrengthLabel;
  delete pEnergyReleaseRateLabel;
  delete pParticleIDLabel;
  delete pIsIgnitedLabel;
  delete pMassRateLabel;
  
  delete pDeformationMeasureLabel_preReloc;
  delete pStressLabel_preReloc;
  delete pVolumeLabel_preReloc;
  delete pMassLabel_preReloc;
  delete pVelocityLabel_preReloc;
  delete pExternalForceLabel_preReloc;
  delete pXLabel_preReloc;
  delete pTemperatureLabel_preReloc;
  delete pTemperatureGradientLabel_preReloc;
  delete pTemperatureRateLabel_preReloc;
  delete pExternalHeatRateLabel_preReloc;
  delete pSurfLabel_preReloc;
  delete pIsBrokenLabel_preReloc;
  delete pCrackNormalLabel_preReloc;
  delete pCrackSurfaceContactForceLabel_preReloc;
  delete pTensileStrengthLabel_preReloc;
  delete pEnergyReleaseRateLabel_preReloc;
  delete pParticleIDLabel_preReloc;
  delete pIsIgnitedLabel_preReloc;
  delete pMassRateLabel_preReloc;

  delete gAccelerationLabel;
  delete gMomExedAccelerationLabel;
  delete gMassLabel;
  delete gVelocityLabel;
  delete gMomExedVelocityLabel;
  delete gExternalForceLabel;
  delete gInternalForceLabel;
  delete gVelocityStarLabel;
  delete gMomExedVelocityStarLabel;
  delete gNormTractionLabel;
  delete gStressLabel;
  delete gSurfNormLabel;
  delete gSelfContactLabel;
  delete gCrackNormalLabel;
  delete gTensileStrengthLabel;
  delete gTemperatureLabel;
  delete gTemperatureStarLabel;
  delete gTemperatureRateLabel;
  delete gInternalHeatRateLabel;
  delete gExternalHeatRateLabel;
  delete gThermalContactHeatExchangeRateLabel;
  delete cBurnedMassLabel;

  delete delTLabel;

  delete StrainEnergyLabel;
  delete KineticEnergyLabel;
  delete TotalMassLabel;
  delete NTractionZMinusLabel;
  delete CenterOfMassPositionLabel;
  delete CenterOfMassVelocityLabel;
  delete ppNAPIDLabel;

  for (int i = 0; i<(int)d_particleState.size(); i++)
    for (int j = 0; j< (int)d_particleState[i].size(); j++)
      delete d_particleState[i][j];

  for (int i = 0; i<(int)d_particleState_preReloc.size(); i++)
    for (int j = 0; j< (int)d_particleState_preReloc[i].size(); j++)
      delete d_particleState_preReloc[i][j];

}

void MPMLabel::registerPermanentParticleState(int i,
					      const VarLabel* label,
					      const VarLabel* preReloc_label)
{
  d_particleState[i].push_back(label);
  d_particleState_preReloc[i].push_back(preReloc_label);
  
}

// $Log$
// Revision 1.48  2001/01/15 22:44:38  tan
// Fixed parallel version of fracture code.
//
// Revision 1.47  2001/01/15 16:34:59  bard
// Added .dat file output invoked using <save label="NTractionZMinus"/>.
// This is the average traction on the z=z_min plane computational
// boundary.  It does not work for multiple patches.
//
// Revision 1.46  2001/01/15 15:54:41  guilkey
// Added mom_L_ME var labels.
//
// Revision 1.45  2001/01/05 23:04:09  guilkey
// Using the code that Wayne just commited which allows the delT variable to
// be "computed" multiple times per timestep, I removed the multiple derivatives
// of delT (delTAfterFracture, delTAfterConstitutiveModel, etc.).  This also
// now allows MPM and ICE to run together with a common timestep.  The
// dream of the sharedState is realized!
//
// Revision 1.44  2000/12/28 20:27:11  guilkey
// Moved some labels from MPMLabel to MPMICELabel.
//
// Revision 1.43  2000/12/01 22:02:47  guilkey
// Made the scheduling of each task a function.  This was done to make
// scheduleTimeAdvance managable, as well as to make it easier to create
// an integrated MPM and CFD code.
//
// Revision 1.42  2000/11/21 20:51:02  tan
// Implemented different models for fracture simulations.  SimpleFracture model
// is for the simulation where the resolution focus only on macroscopic major
// cracks. NormalFracture and ExplosionFracture models are more sophiscated
// and specific fracture models that are currently underconstruction.
//
// Revision 1.41  2000/09/25 20:23:13  sparker
// Quiet g++ warnings
//
// Revision 1.40  2000/09/22 07:14:06  tan
// MPM code works with fracture in three point bending.
//
// Revision 1.39  2000/09/16 04:27:34  tan
// Modifications to make fracture works well.
//
// Revision 1.38  2000/09/11 20:17:29  tan
// Fixed a typo of crackSurfaceContactForce.
//
// Revision 1.37  2000/09/11 18:56:25  tan
// Crack surface contact force is now considered in the simulation.
//
// Revision 1.36  2000/09/11 03:12:20  tan
// Added energy release rate computations for fracture.
//
// Revision 1.35  2000/09/11 01:08:37  tan
// Modified time step calculation (in constitutive model computeStressTensor(...))
// when fracture cracking speed involved.
//
// Revision 1.34  2000/09/11 00:14:55  tan
// Added calculations on random distributed microcracks in broken particles.
//
// Revision 1.33  2000/09/10 22:51:09  tan
// Added particle rotationRate computation in computeStressTensor functions
// in each constitutive model classes.  The particle rotationRate will be used
// for fracture.
//
// Revision 1.32  2000/09/09 19:34:11  tan
// Added MPMLabel::pVisibilityLabel and SerialMPM::computerNodesVisibility().
//
// Revision 1.31  2000/09/08 20:27:59  tan
// Added visibility calculation to fracture broken cell shape function
// interpolation.
//
// Revision 1.30  2000/09/08 17:31:28  guilkey
// Added interpolateParticlesForSaving task which interpolates particle
// data, interpolates it to the grid using another particle scalar variable
// for weighting, and saves it to the grid data to the uda.  Note that these
// interpolations only get done when it's time to save data to the uda.
//
// Revision 1.29  2000/09/08 01:47:34  tan
// Added pDilationalWaveSpeedLabel for fracture and is saved as a
// side-effect of computeStressTensor in each constitutive model class.
//
// Revision 1.28  2000/09/07 21:11:04  tan
// Added particle variable pMicrocrackSize for fracture.
//
// Revision 1.27  2000/09/05 05:16:02  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.26  2000/08/30 00:12:42  guilkey
// Added some stuff for interpolating particle data to the grid solely
// for the purpose of saving to an uda.  This doesn't work yet.
//
// Revision 1.25  2000/08/09 03:17:58  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.24  2000/08/08 20:00:32  tan
// Added cCrackedCellLabel and pCrackNormalLabel for fracture.
//
// Revision 1.23  2000/08/08 01:32:41  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.22  2000/08/04 16:42:43  guilkey
// Added VarLabels specific to FrictionContact so that those variables
// can be pleaseSaved.
//
// Revision 1.21  2000/07/27 22:17:16  jas
// Consolidated the registerPermanentParticleState to take both the
// regular labels and the pre_Reloc labels.
//
// Revision 1.20  2000/07/27 20:29:50  jas
// In SerialMPM.cc, problemSetup, there are now labels for each material.
// So it is now possible for different materials to have different VarLabels
// depending on the simulation requirements.
//
// Revision 1.19  2000/07/17 23:41:33  tan
// Fixed problems in MPM heat conduction.
//
// Revision 1.18  2000/07/12 18:45:06  jas
// Cleaned up the creation of the particle state and moved things into
// MPMLabel.
//
// Revision 1.17  2000/07/05 23:43:29  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.16  2000/06/28 01:09:06  tan
// Thermal contact model start to work!
//
// Revision 1.15  2000/06/27 21:50:57  guilkey
// Added saving of more "meta data."  Namely, center of mass position and
// velocity.  These are both actually mass weighted, so should be divided
// by total mass which is being saved in TotalMass.dat.
//
// Revision 1.14  2000/06/23 21:18:34  tan
// Added pExternalHeatRateLabel for heat conduction.
//
// Revision 1.13  2000/06/23 20:56:14  tan
// Fixed mistakes in label names of gInternalHeatRateLabel and
// gExternalHeatRateLabel.
//
// Revision 1.12  2000/06/23 20:02:26  tan
// Create pTemperatureLabel in the MPMLable constructor.
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
// Revision 1.7  2000/06/03 05:25:43  sparker
// Added a new for pSurfLabel (was uninitialized)
// Uncommented pleaseSaveIntegrated
// Minor cleanups of reduction variable use
// Removed a few warnings
//
// Revision 1.6  2000/06/02 23:16:32  guilkey
// Added ParticleID labels.
//
// Revision 1.5  2000/05/31 22:15:38  guilkey
// Added VarLabels for some integrated quantities.
//
// Revision 1.4  2000/05/31 16:30:57  guilkey
// Tidied the file up, added cvs logging.
//
