#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

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

  pCrackSurfaceNormalLabel = scinew VarLabel( "p.crackSurfaceNormal",
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

  pCrackSurfaceNormalLabel_preReloc = scinew VarLabel( "p.crackSurfaceNormal+",
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
  cVelocityLabel = scinew VarLabel( "c.velocity",
			CCVariable<Vector>::getTypeDescription() );
  cMassLabel = scinew VarLabel( "c.mass",
			CCVariable<double>::getTypeDescription() );

  // Reduction variables

  delTAfterConstitutiveModelLabel = scinew VarLabel( 
    "delTAfterConstitutiveModel", 
    delt_vartype::getTypeDescription() );

  delTAfterFractureLabel = scinew VarLabel( "delTAfterFracture", 
    delt_vartype::getTypeDescription() );

  delTAfterCrackSurfaceContactLabel = scinew VarLabel( 
    "delTAfterCrackSurafceContact", 
    delt_vartype::getTypeDescription() );

  delTLabel = scinew VarLabel( "delT", delt_vartype::getTypeDescription() );

  StrainEnergyLabel = scinew VarLabel( "StrainEnergy",
			sum_vartype::getTypeDescription() );

  KineticEnergyLabel = scinew VarLabel( "KineticEnergy",
			sum_vartype::getTypeDescription() );

  TotalMassLabel = scinew VarLabel( "TotalMass",
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
  delete pCrackSurfaceNormalLabel;
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
  delete pCrackSurfaceNormalLabel_preReloc;
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
  delete gTemperatureLabel;
  delete gTemperatureStarLabel;
  delete gTemperatureRateLabel;
  delete gInternalHeatRateLabel;
  delete gExternalHeatRateLabel;
  delete gThermalContactHeatExchangeRateLabel;
  delete cBurnedMassLabel;
  delete cVelocityLabel;
  delete cMassLabel;

  delete delTAfterConstitutiveModelLabel;
  delete delTAfterFractureLabel;
  delete delTAfterCrackSurfaceContactLabel;
  delete delTLabel;

  delete StrainEnergyLabel;
  delete KineticEnergyLabel;
  delete TotalMassLabel;
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

