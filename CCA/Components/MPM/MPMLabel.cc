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
  pDeformationMeasureLabel = VarLabel::create("p.deformationMeasure",
			ParticleVariable<Matrix3>::getTypeDescription());

  pCrackEffectiveLabel = VarLabel::create("p.crackEffective",
			ParticleVariable<int>::getTypeDescription());

  pConnectivityLabel = VarLabel::create("p.connectivity",
			ParticleVariable<int>::getTypeDescription());

  pContactForceLabel = VarLabel::create("p.contactForce",
			ParticleVariable<Vector>::getTypeDescription());

  pXXLabel = VarLabel::create("p.positionXX",
			ParticleVariable<Point>::getTypeDescription());
  
  pStressLabel_afterFracture = VarLabel::create(
                        "p.stress_afterFracture",
			ParticleVariable<Matrix3>::getTypeDescription());

  pStressLabel_afterStrainRate = VarLabel::create(
                        "p.stress_afterStrainRate",
			ParticleVariable<Matrix3>::getTypeDescription());

  pVelocityLabel_afterFracture = VarLabel::create(
                        "p.velocity_afterFracture",
			ParticleVariable<Vector>::getTypeDescription());

  pVelocityLabel_afterUpdate = VarLabel::create(
                        "p.velocity_afterUpdate",
			ParticleVariable<Vector>::getTypeDescription());

  pStrainEnergyLabel = VarLabel::create("p.strainEnergy",
			ParticleVariable<double>::getTypeDescription());

  pRotationRateLabel = VarLabel::create("p.rotationRate",
			ParticleVariable<Vector>::getTypeDescription());

  pPressureLabel  = VarLabel::create( "p.pressure",
			ParticleVariable<double>::getTypeDescription() );
  
  //PermanentParticleState
  pStressLabel = VarLabel::create( "p.stress",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pCrackRadiusLabel = VarLabel::create( "p.CrackRadius",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeLabel = VarLabel::create( "p.volume",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeDeformedLabel = VarLabel::create( "p.volumedeformed",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel = VarLabel::create( "p.mass",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel = VarLabel::create( "p.velocity", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel = VarLabel::create( "p.externalforce",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel = VarLabel::create("p.x",ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);
  
  pTemperatureLabel = VarLabel::create( "p.temperature",
			ParticleVariable<double>::getTypeDescription() );
  
  pTemperatureGradientLabel = VarLabel::create( "p.temperatureGradient",
			ParticleVariable<Vector>::getTypeDescription() );

  pExternalHeatRateLabel = VarLabel::create( "p.externalHeatRate",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel = VarLabel::create( "p.surface",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel = VarLabel::create( "p.isBroken",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel = VarLabel::create( "p.crackNormal",
			ParticleVariable<Vector>::getTypeDescription() );

  pTipNormalLabel = VarLabel::create( "p.tipNormal",
			ParticleVariable<Vector>::getTypeDescription() );

  pExtensionDirectionLabel = VarLabel::create( "p.extensionDirection",
			ParticleVariable<Vector>::getTypeDescription() );

  pToughnessLabel = VarLabel::create( "p.toughness",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel = VarLabel::create( "p.energyReleaseRate",
			ParticleVariable<double>::getTypeDescription() );

  pCrackSurfacePressureLabel = VarLabel::create( "p.crackSurfacePressure",
			ParticleVariable<double>::getTypeDescription() );

  pDisplacementLabel = VarLabel::create( "p.displacement",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel = VarLabel::create("p.particleID",
			ParticleVariable<long64>::getTypeDescription() );

  pIsIgnitedLabel  = VarLabel::create( "p.isIgnited",
			ParticleVariable<int>::getTypeDescription() );
  
  pMassRateLabel  = VarLabel::create( "p.massRate",
			ParticleVariable<double>::getTypeDescription() );
  
  pTang1Label  = VarLabel::create( "p.tang1",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang2Label  = VarLabel::create( "p.tang2",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang1Label_preReloc  = VarLabel::create( "p.tang1+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang2Label_preReloc  = VarLabel::create( "p.tang2+",
                        ParticleVariable<Vector>::getTypeDescription() );

  // Particle Variables 
  pDeformationMeasureLabel_preReloc = VarLabel::create("p.deformationMeasure+",
			ParticleVariable<Matrix3>::getTypeDescription());
  
  pStressLabel_preReloc = VarLabel::create( "p.stress+",
			ParticleVariable<Matrix3>::getTypeDescription() );

  pCrackRadiusLabel_preReloc = VarLabel::create( "p.CrackRadius+",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeLabel_preReloc = VarLabel::create( "p.volume+",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel_preReloc = VarLabel::create( "p.mass+",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel_preReloc = VarLabel::create( "p.velocity+", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel_preReloc = VarLabel::create( "p.externalforce+",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel_preReloc = VarLabel::create( "p.x+",
			ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);
  
  pTemperatureLabel_preReloc = VarLabel::create( "p.temperature+",
			ParticleVariable<double>::getTypeDescription() );
  
  pExternalHeatRateLabel_preReloc = VarLabel::create( "p.externalHeatRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel_preReloc = VarLabel::create( "p.surface+",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel_preReloc = VarLabel::create( "p.isBroken+",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel_preReloc = VarLabel::create( "p.crackNormal+",
			ParticleVariable<Vector>::getTypeDescription() );

  pTipNormalLabel_preReloc = VarLabel::create( "p.tipNormal+",
			ParticleVariable<Vector>::getTypeDescription() );

  pExtensionDirectionLabel_preReloc = VarLabel::create( "p.extensionDirection+",
			ParticleVariable<Vector>::getTypeDescription() );

  pToughnessLabel_preReloc = VarLabel::create( "p.toughness+",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel_preReloc =VarLabel::create("p.energyReleaseRate+",
			ParticleVariable<double>::getTypeDescription() );

  pCrackSurfacePressureLabel_preReloc = 
                        VarLabel::create( "p.crackSurfacePressure+",
                        ParticleVariable<double>::getTypeDescription() );

  pDisplacementLabel_preReloc = VarLabel::create( "p.displacement+",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+",
			ParticleVariable<long64>::getTypeDescription() );

  pIsIgnitedLabel_preReloc  = VarLabel::create( "p.isIgnited+",
			ParticleVariable<int>::getTypeDescription() );
  
  pMassRateLabel_preReloc  = VarLabel::create( "p.massRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  // Node Centered Variables
  
  gAccelerationLabel = VarLabel::create( "g.acceleration",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedAccelerationLabel = VarLabel::create( "g.momexedacceleration",
			NCVariable<Vector>::getTypeDescription() );
  
  gMassLabel = VarLabel::create( "g.mass",
			NCVariable<double>::getTypeDescription() );
  
  gVelocityLabel = VarLabel::create( "g.velocity",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityLabel = VarLabel::create( "g.momexedvelocity",
			NCVariable<Vector>::getTypeDescription() );
  
  gExternalForceLabel = VarLabel::create( "g.externalforce",
			NCVariable<Vector>::getTypeDescription() );
  
  gInternalForceLabel = VarLabel::create( "g.internalforce",
			NCVariable<Vector>::getTypeDescription() );
  
  gVelocityStarLabel = VarLabel::create( "g.velocity_star",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityStarLabel = VarLabel::create( "g.momexedvelocity_star",
			NCVariable<Vector>::getTypeDescription() );
  
  gTemperatureLabel = VarLabel::create("g.temperature",
			NCVariable<double>::getTypeDescription());

  gTemperatureNoBCLabel = VarLabel::create("g.temperaturenobc",
			NCVariable<double>::getTypeDescription());

  gTemperatureStarLabel = VarLabel::create("g.temperatureStar",
			NCVariable<double>::getTypeDescription());

  gTemperatureRateLabel = VarLabel::create("g.temperatureRate",
			NCVariable<double>::getTypeDescription());

  gInternalHeatRateLabel = VarLabel::create("g.internalHeatRate",
			NCVariable<double>::getTypeDescription());

  gExternalHeatRateLabel = VarLabel::create("g.externalHeatRate",
			NCVariable<double>::getTypeDescription());

  gThermalContactHeatExchangeRateLabel = 
     VarLabel::create("g.thermalContactHeatExchangeRate",
     NCVariable<double>::getTypeDescription());

  gNormTractionLabel = VarLabel::create( "g.normtraction",
                   NCVariable<double>::getTypeDescription() );

  gSurfNormLabel = VarLabel::create( "g.surfnorm",
                   NCVariable<Vector>::getTypeDescription() );

  gStressLabel   = VarLabel::create( "g.stress",
                   NCVariable<Matrix3>::getTypeDescription() );

  gStressForSavingLabel   = VarLabel::create( "g.stressFS",
                   NCVariable<Matrix3>::getTypeDescription() );

  gVolumeLabel     = VarLabel::create("g.volume",
			NCVariable<double>::getTypeDescription());

  gWeightLabel     = VarLabel::create("g.weight",
			NCVariable<double>::getTypeDescription());

  gradPAccNCLabel = VarLabel::create("gradPAccNC",
			NCVariable<Vector>::getTypeDescription());

  dTdt_NCLabel     = VarLabel::create("dTdt_NC",
			NCVariable<double>::getTypeDescription());

  massBurnFractionLabel  = VarLabel::create("massBurnFraction",
			NCVariable<double>::getTypeDescription());

  AccArchesNCLabel = VarLabel::create("AccArchesNC",
			NCVariable<Vector>::getTypeDescription() );

  frictionalWorkLabel = VarLabel::create("frictionalWork",
			NCVariable<double>::getTypeDescription());

  // Reduction variables
  partCountLabel = VarLabel::create("particleCount",
				   sumlong_vartype::getTypeDescription());

  delTLabel = VarLabel::create( "delT", delt_vartype::getTypeDescription() );

  StrainEnergyLabel = VarLabel::create( "StrainEnergy",
			sum_vartype::getTypeDescription() );

  KineticEnergyLabel = VarLabel::create( "KineticEnergy",
			sum_vartype::getTypeDescription() );

  TotalMassLabel = VarLabel::create( "TotalMass",
				 sum_vartype::getTypeDescription() );

  NTractionZMinusLabel = VarLabel::create( "NTractionZMinus",
			sum_vartype::getTypeDescription() );

  CenterOfMassPositionLabel = VarLabel::create( "CenterOfMassPosition",
				 sumvec_vartype::getTypeDescription() );

  CenterOfMassVelocityLabel = VarLabel::create( "CenterOfMassVelocity",
				 sumvec_vartype::getTypeDescription() );

  // for assigning particle ids
  pCellNAPIDLabel =
    VarLabel::create("cellNAPID", CCVariable<short int>::getTypeDescription());

  doMechLabel = VarLabel::create( "doMech", delt_vartype::getTypeDescription() );

} 

MPMLabel::~MPMLabel()
{
  //non PermanentParticleState
  VarLabel::destroy(pDeformationMeasureLabel);
  VarLabel::destroy(pConnectivityLabel);
  VarLabel::destroy(pCrackEffectiveLabel);
  VarLabel::destroy(pContactForceLabel);
  VarLabel::destroy(pXXLabel);

  VarLabel::destroy(pStressLabel_afterFracture);
  VarLabel::destroy(pStressLabel_afterStrainRate);
  VarLabel::destroy(pVelocityLabel_afterFracture);
  VarLabel::destroy(pVelocityLabel_afterUpdate);

  VarLabel::destroy(pStrainEnergyLabel);
  VarLabel::destroy(pRotationRateLabel);

  //PermanentParticleState
  VarLabel::destroy(pStressLabel);
  VarLabel::destroy(pVolumeLabel);
  VarLabel::destroy(pVolumeDeformedLabel);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pVelocityLabel);
  VarLabel::destroy(pExternalForceLabel);
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pTemperatureLabel);
  VarLabel::destroy(pTemperatureGradientLabel);
  VarLabel::destroy(pExternalHeatRateLabel);
  VarLabel::destroy(pSurfLabel);
  VarLabel::destroy(pIsBrokenLabel);
  VarLabel::destroy(pCrackNormalLabel);
  VarLabel::destroy(pTipNormalLabel);
  VarLabel::destroy(pExtensionDirectionLabel);
  VarLabel::destroy(pToughnessLabel);
  VarLabel::destroy(pEnergyReleaseRateLabel);
  VarLabel::destroy(pCrackSurfacePressureLabel);
  VarLabel::destroy(pDisplacementLabel);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pIsIgnitedLabel);
  VarLabel::destroy(pMassRateLabel);
  VarLabel::destroy(pPressureLabel);
  VarLabel::destroy(pCrackRadiusLabel);
  VarLabel::destroy(pTang1Label);
  VarLabel::destroy(pTang2Label);
  VarLabel::destroy(pTang1Label_preReloc);
  VarLabel::destroy(pTang2Label_preReloc);
  
  VarLabel::destroy(pDeformationMeasureLabel_preReloc);
  VarLabel::destroy(pStressLabel_preReloc);
  VarLabel::destroy(pVolumeLabel_preReloc);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pVelocityLabel_preReloc);
  VarLabel::destroy(pExternalForceLabel_preReloc);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pTemperatureLabel_preReloc);
  VarLabel::destroy(pExternalHeatRateLabel_preReloc);
  VarLabel::destroy(pSurfLabel_preReloc);
  VarLabel::destroy(pIsBrokenLabel_preReloc);
  VarLabel::destroy(pCrackNormalLabel_preReloc);
  VarLabel::destroy(pTipNormalLabel_preReloc);
  VarLabel::destroy(pExtensionDirectionLabel_preReloc);
  VarLabel::destroy(pToughnessLabel_preReloc);
  VarLabel::destroy(pEnergyReleaseRateLabel_preReloc);
  VarLabel::destroy(pCrackSurfacePressureLabel_preReloc);
  VarLabel::destroy(pDisplacementLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel_preReloc);
  VarLabel::destroy(pIsIgnitedLabel_preReloc);
  VarLabel::destroy(pMassRateLabel_preReloc);
  VarLabel::destroy(pCrackRadiusLabel_preReloc);

  VarLabel::destroy(gAccelerationLabel);
  VarLabel::destroy(gMomExedAccelerationLabel);
  VarLabel::destroy(gMassLabel);
  VarLabel::destroy(gVelocityLabel);
  VarLabel::destroy(gMomExedVelocityLabel);
  VarLabel::destroy(gExternalForceLabel);
  VarLabel::destroy(gInternalForceLabel);
  VarLabel::destroy(gVelocityStarLabel);
  VarLabel::destroy(gMomExedVelocityStarLabel);
  VarLabel::destroy(gNormTractionLabel);
  VarLabel::destroy(gStressLabel);
  VarLabel::destroy(gSurfNormLabel);
  VarLabel::destroy(gTemperatureLabel);
  VarLabel::destroy(gTemperatureNoBCLabel);
  VarLabel::destroy(gTemperatureStarLabel);
  VarLabel::destroy(gTemperatureRateLabel);
  VarLabel::destroy(gInternalHeatRateLabel);
  VarLabel::destroy(gExternalHeatRateLabel);
  VarLabel::destroy(gThermalContactHeatExchangeRateLabel);
  VarLabel::destroy(gStressForSavingLabel);
  VarLabel::destroy(gVolumeLabel);
  VarLabel::destroy(gWeightLabel);
  VarLabel::destroy(gradPAccNCLabel);
  VarLabel::destroy(dTdt_NCLabel);
  VarLabel::destroy(massBurnFractionLabel);
  VarLabel::destroy(AccArchesNCLabel);
  VarLabel::destroy(frictionalWorkLabel);

  VarLabel::destroy(partCountLabel);
  VarLabel::destroy(delTLabel);
  VarLabel::destroy(doMechLabel);

  VarLabel::destroy(StrainEnergyLabel);
  VarLabel::destroy(KineticEnergyLabel);
  VarLabel::destroy(TotalMassLabel);
  VarLabel::destroy(NTractionZMinusLabel);
  VarLabel::destroy(CenterOfMassPositionLabel);
  VarLabel::destroy(CenterOfMassVelocityLabel);
  VarLabel::destroy(pCellNAPIDLabel);
}

void MPMLabel::registerPermanentParticleState(int i,
					      const VarLabel* label,
					      const VarLabel* preReloc_label)
{
  d_particleState[i].push_back(label);
  d_particleState_preReloc[i].push_back(preReloc_label);
  
}
