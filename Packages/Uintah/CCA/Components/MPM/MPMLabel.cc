#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
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

  pCrackEffectiveLabel = scinew VarLabel("p.crackEffective",
			ParticleVariable<int>::getTypeDescription());

  pConnectivityLabel = scinew VarLabel("p.connectivity",
			ParticleVariable<int>::getTypeDescription());

  pContactForceLabel = scinew VarLabel("p.contactForce",
			ParticleVariable<Vector>::getTypeDescription());

  pXXLabel = scinew VarLabel("p.positionXX",
			ParticleVariable<Point>::getTypeDescription());
  
  pStressLabel_afterFracture = scinew VarLabel(
                        "p.stress_afterFracture",
			ParticleVariable<Matrix3>::getTypeDescription());

  pStressLabel_afterStrainRate = scinew VarLabel(
                        "p.stress_afterStrainRate",
			ParticleVariable<Matrix3>::getTypeDescription());

  pVelocityLabel_afterFracture = scinew VarLabel(
                        "p.velocity_afterFracture",
			ParticleVariable<Vector>::getTypeDescription());

  pVelocityLabel_afterUpdate = scinew VarLabel(
                        "p.velocity_afterUpdate",
			ParticleVariable<Vector>::getTypeDescription());

  pStrainEnergyLabel = scinew VarLabel("p.strainEnergy",
			ParticleVariable<double>::getTypeDescription());

  pRotationRateLabel = scinew VarLabel("p.rotationRate",
			ParticleVariable<Vector>::getTypeDescription());

  pPressureLabel  = scinew VarLabel( "p.pressure",
			ParticleVariable<double>::getTypeDescription() );
  
  //PermanentParticleState
  pStressLabel = scinew VarLabel( "p.stress",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pCrackRadiusLabel = scinew VarLabel( "p.CrackRadius",
			ParticleVariable<double>::getTypeDescription());
  
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

  pExternalHeatRateLabel = scinew VarLabel( "p.externalHeatRate",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel = scinew VarLabel( "p.surface",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel = scinew VarLabel( "p.isBroken",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel = scinew VarLabel( "p.crackNormal",
			ParticleVariable<Vector>::getTypeDescription() );

  pTipNormalLabel = scinew VarLabel( "p.tipNormal",
			ParticleVariable<Vector>::getTypeDescription() );

  pExtensionDirectionLabel = scinew VarLabel( "p.extensionDirection",
			ParticleVariable<Vector>::getTypeDescription() );

  pToughnessLabel = scinew VarLabel( "p.toughness",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel = scinew VarLabel( "p.energyReleaseRate",
			ParticleVariable<double>::getTypeDescription() );

  pCrackSurfacePressureLabel = scinew VarLabel( "p.crackSurfacePressure",
			ParticleVariable<double>::getTypeDescription() );

  pDisplacementLabel = scinew VarLabel( "p.displacement",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel = scinew VarLabel("p.particleID",
			ParticleVariable<long64>::getTypeDescription() );

  pIsIgnitedLabel  = scinew VarLabel( "p.isIgnited",
			ParticleVariable<int>::getTypeDescription() );
  
  pMassRateLabel  = scinew VarLabel( "p.massRate",
			ParticleVariable<double>::getTypeDescription() );
  
  // Particle Variables 
  pDeformationMeasureLabel_preReloc = scinew VarLabel("p.deformationMeasure+",
			ParticleVariable<Matrix3>::getTypeDescription());
  
  pStressLabel_preReloc = scinew VarLabel( "p.stress+",
			ParticleVariable<Matrix3>::getTypeDescription() );

  pCrackRadiusLabel_preReloc = scinew VarLabel( "p.CrackRadius+",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeLabel_preReloc = scinew VarLabel( "p.volume+",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel_preReloc = scinew VarLabel( "p.mass+",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel_preReloc = scinew VarLabel( "p.velocity+", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel_preReloc = scinew VarLabel( "p.externalforce+",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel_preReloc = scinew VarLabel( "p.x+",
			ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);
  
  pTemperatureLabel_preReloc = scinew VarLabel( "p.temperature+",
			ParticleVariable<double>::getTypeDescription() );
  
  pExternalHeatRateLabel_preReloc = scinew VarLabel( "p.externalHeatRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel_preReloc = scinew VarLabel( "p.surface+",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel_preReloc = scinew VarLabel( "p.isBroken+",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel_preReloc = scinew VarLabel( "p.crackNormal+",
			ParticleVariable<Vector>::getTypeDescription() );

  pTipNormalLabel_preReloc = scinew VarLabel( "p.tipNormal+",
			ParticleVariable<Vector>::getTypeDescription() );

  pExtensionDirectionLabel_preReloc = scinew VarLabel( "p.extensionDirection+",
			ParticleVariable<Vector>::getTypeDescription() );

  pToughnessLabel_preReloc = scinew VarLabel( "p.toughness+",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel_preReloc =scinew VarLabel("p.energyReleaseRate+",
			ParticleVariable<double>::getTypeDescription() );

  pCrackSurfacePressureLabel_preReloc = scinew 
                        VarLabel( "p.crackSurfacePressure+",
                        ParticleVariable<double>::getTypeDescription() );

  pDisplacementLabel_preReloc = scinew VarLabel( "p.displacement+",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel_preReloc = scinew VarLabel("p.particleID+",
			ParticleVariable<long64>::getTypeDescription() );

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
  
  gTemperatureLabel = scinew VarLabel("g.temperature",
			NCVariable<double>::getTypeDescription());

  gTemperatureNoBCLabel = scinew VarLabel("g.temperaturenobc",
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

  gVolumeLabel     = scinew VarLabel("g.volume",
			NCVariable<double>::getTypeDescription());

  gWeightLabel     = scinew VarLabel("g.weight",
			NCVariable<double>::getTypeDescription());

  gradPAccNCLabel = scinew VarLabel("gradPAccNC",
			NCVariable<Vector>::getTypeDescription());

  dTdt_NCLabel     = scinew VarLabel("dTdt_NC",
			NCVariable<double>::getTypeDescription());

  massBurnFractionLabel  = scinew VarLabel("massBurnFraction",
			NCVariable<double>::getTypeDescription());

  AccArchesNCLabel = scinew VarLabel("AccArchesNC",
			NCVariable<Vector>::getTypeDescription() );

  frictionalWorkLabel = scinew VarLabel("frictionalWork",
			NCVariable<double>::getTypeDescription());

  // Reduction variables
  partCountLabel = scinew VarLabel("particleCount",
				   sumlong_vartype::getTypeDescription());

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

  // for assigning particle ids
  pCellNAPIDLabel =
    scinew VarLabel("cellNAPID", CCVariable<short int>::getTypeDescription());

  doMechLabel = scinew VarLabel( "doMech", delt_vartype::getTypeDescription() );

} 

MPMLabel::~MPMLabel()
{
  //non PermanentParticleState
  delete pDeformationMeasureLabel;
  delete pConnectivityLabel;
  delete pCrackEffectiveLabel;
  delete pContactForceLabel;
  delete pXXLabel;

  delete pStressLabel_afterFracture;
  delete pStressLabel_afterStrainRate;
  delete pVelocityLabel_afterFracture;
  delete pVelocityLabel_afterUpdate;

  delete pStrainEnergyLabel;
  delete pRotationRateLabel;

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
  delete pExternalHeatRateLabel;
  delete pSurfLabel;
  delete pIsBrokenLabel;
  delete pCrackNormalLabel;
  delete pTipNormalLabel;
  delete pExtensionDirectionLabel;
  delete pToughnessLabel;
  delete pEnergyReleaseRateLabel;
  delete pCrackSurfacePressureLabel;
  delete pDisplacementLabel;
  delete pParticleIDLabel;
  delete pIsIgnitedLabel;
  delete pMassRateLabel;
  delete pPressureLabel;
  delete pCrackRadiusLabel;
  
  delete pDeformationMeasureLabel_preReloc;
  delete pStressLabel_preReloc;
  delete pVolumeLabel_preReloc;
  delete pMassLabel_preReloc;
  delete pVelocityLabel_preReloc;
  delete pExternalForceLabel_preReloc;
  delete pXLabel_preReloc;
  delete pTemperatureLabel_preReloc;
  delete pExternalHeatRateLabel_preReloc;
  delete pSurfLabel_preReloc;
  delete pIsBrokenLabel_preReloc;
  delete pCrackNormalLabel_preReloc;
  delete pTipNormalLabel_preReloc;
  delete pExtensionDirectionLabel_preReloc;
  delete pToughnessLabel_preReloc;
  delete pEnergyReleaseRateLabel_preReloc;
  delete pCrackSurfacePressureLabel_preReloc;
  delete pDisplacementLabel_preReloc;
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
  delete gTemperatureLabel;
  delete gTemperatureNoBCLabel;
  delete gTemperatureStarLabel;
  delete gTemperatureRateLabel;
  delete gInternalHeatRateLabel;
  delete gExternalHeatRateLabel;
  delete gThermalContactHeatExchangeRateLabel;
  delete gStressForSavingLabel;
  delete gVolumeLabel;
  delete gWeightLabel;
  delete gradPAccNCLabel;
  delete dTdt_NCLabel;
  delete massBurnFractionLabel;
  delete AccArchesNCLabel;
  delete frictionalWorkLabel;

  delete partCountLabel;
  delete delTLabel;
  delete doMechLabel;

  delete StrainEnergyLabel;
  delete KineticEnergyLabel;
  delete TotalMassLabel;
  delete NTractionZMinusLabel;
  delete CenterOfMassPositionLabel;
  delete CenterOfMassVelocityLabel;
  delete pCellNAPIDLabel;
}

void MPMLabel::registerPermanentParticleState(int i,
					      const VarLabel* label,
					      const VarLabel* preReloc_label)
{
  d_particleState[i].push_back(label);
  d_particleState_preReloc[i].push_back(preReloc_label);
  
}
