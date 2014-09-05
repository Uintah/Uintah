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
  pPressureLabel  = VarLabel::create( "p.pressure",
			ParticleVariable<double>::getTypeDescription() );
  
  pTemperatureGradientLabel = VarLabel::create( "p.temperatureGradient",
			ParticleVariable<Vector>::getTypeDescription() );

  //PermanentParticleState
  pDeformationMeasureLabel = VarLabel::create("p.deformationMeasure",
			ParticleVariable<Matrix3>::getTypeDescription());

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
  
  pXLabel = VarLabel::create("p.x",
    ParticleVariable<Point>::getTypeDescription(),VarLabel::PositionVariable);
  
  pTemperatureLabel = VarLabel::create( "p.temperature",
			ParticleVariable<double>::getTypeDescription() );
  
  pExternalHeatRateLabel = VarLabel::create( "p.externalHeatRate",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel = VarLabel::create( "p.surface",
			ParticleVariable<int>::getTypeDescription() );

  pParticleIDLabel = VarLabel::create("p.particleID",
			ParticleVariable<long64>::getTypeDescription() );

  pSizeLabel = VarLabel::create( "p.size",
			ParticleVariable<Vector>::getTypeDescription());
  
  pTang1Label  = VarLabel::create( "p.tang1",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang2Label  = VarLabel::create( "p.tang2",
                        ParticleVariable<Vector>::getTypeDescription() );

  pNormLabel  = VarLabel::create( "p.norm",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang1Label_preReloc  = VarLabel::create( "p.tang1+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang2Label_preReloc  = VarLabel::create( "p.tang2+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pNormLabel_preReloc  = VarLabel::create( "p.norm+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pSizeLabel_preReloc = VarLabel::create( "p.size+",
			ParticleVariable<Vector>::getTypeDescription());
  
  // Particle Variables 
  pDeformationMeasureLabel_preReloc = VarLabel::create("p.deformationMeasure+",
			ParticleVariable<Matrix3>::getTypeDescription());
  
  pStressLabel_preReloc = VarLabel::create( "p.stress+",
			ParticleVariable<Matrix3>::getTypeDescription() );

  pStress_eLabel = VarLabel::create( "p.stress_e",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pStress_veLabel = VarLabel::create( "p.stress_ve",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pStress_ve_vLabel = VarLabel::create( "p.stress_ve_v",
			ParticleVariable<double>::getTypeDescription() );
			
  pStress_ve_dLabel = VarLabel::create( "p.stress_ve_d",
			ParticleVariable<Matrix3>::getTypeDescription() );
			
  pStress_e_vLabel = VarLabel::create( "p.stress_e_v",
			ParticleVariable<double>::getTypeDescription() );
			
  pStress_e_dLabel = VarLabel::create( "p.stress_e_d",
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

  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+",
			ParticleVariable<long64>::getTypeDescription() );

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

  gNumNearParticlesLabel = VarLabel::create("NumNearParticles",
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

  doMechLabel = VarLabel::create( "doMech", delt_vartype::getTypeDescription());

  // Implicit MPM labels

  dispNewLabel = VarLabel::create("dispNew",
				  NCVariable<Vector>::getTypeDescription()); 

  dispIncLabel = VarLabel::create("dispInc",
				  NCVariable<Vector>::getTypeDescription()); 

  dispIncQNorm0 = VarLabel::create("dispIncQNorm0",
				   sum_vartype::getTypeDescription());

  dispIncNormMax = VarLabel::create("dispIncNormMax",
				    sum_vartype::getTypeDescription());

  dispIncQNorm = VarLabel::create("dispIncQNorm",
				  sum_vartype::getTypeDescription());

  dispIncNorm = VarLabel::create("dispIncNorm",
				 sum_vartype::getTypeDescription());
  
  pAccelerationLabel = VarLabel::create("p.acceleration",
				   ParticleVariable<Vector>::getTypeDescription()); 

  pAccelerationLabel_preReloc = VarLabel::create("p.acceleration+",
				   ParticleVariable<Vector>::getTypeDescription()); 

  bElBarLabel = VarLabel::create("bElBar",
		       ParticleVariable<Matrix3>::getTypeDescription());

  bElBarLabel_preReloc = VarLabel::create("bElBar_preReloc",
			 ParticleVariable<Matrix3>::getTypeDescription());


} 

MPMLabel::~MPMLabel()
{
  //non PermanentParticleState
  VarLabel::destroy(pVolumeDeformedLabel);
  VarLabel::destroy(pTemperatureGradientLabel);

  //PermanentParticleState
  VarLabel::destroy(pDeformationMeasureLabel);
  VarLabel::destroy(pDeformationMeasureLabel_preReloc);
  VarLabel::destroy(pStressLabel);
  VarLabel::destroy(pStressLabel_preReloc);
  VarLabel::destroy(pStress_eLabel);
  VarLabel::destroy(pStress_veLabel);
  VarLabel::destroy(pStress_ve_vLabel);
  VarLabel::destroy(pStress_ve_dLabel);
  VarLabel::destroy(pStress_e_vLabel);
  VarLabel::destroy(pStress_e_dLabel);
  VarLabel::destroy(pVolumeLabel);
  VarLabel::destroy(pVolumeLabel_preReloc);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pVelocityLabel);
  VarLabel::destroy(pVelocityLabel_preReloc);
  VarLabel::destroy(pExternalForceLabel);
  VarLabel::destroy(pExternalForceLabel_preReloc);
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pTemperatureLabel);
  VarLabel::destroy(pTemperatureLabel_preReloc);
  VarLabel::destroy(pExternalHeatRateLabel);
  VarLabel::destroy(pExternalHeatRateLabel_preReloc);
  VarLabel::destroy(pSurfLabel);
  VarLabel::destroy(pSurfLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);
  VarLabel::destroy(pPressureLabel);
  VarLabel::destroy(pCrackRadiusLabel);
  VarLabel::destroy(pCrackRadiusLabel_preReloc);
  VarLabel::destroy(pSizeLabel);
  VarLabel::destroy(pSizeLabel_preReloc);
  VarLabel::destroy(pTang1Label);
  VarLabel::destroy(pTang1Label_preReloc);
  VarLabel::destroy(pTang2Label);
  VarLabel::destroy(pTang2Label_preReloc);
  VarLabel::destroy(pNormLabel);
  VarLabel::destroy(pNormLabel_preReloc);
  

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
  VarLabel::destroy(gradPAccNCLabel);
  VarLabel::destroy(dTdt_NCLabel);
  VarLabel::destroy(massBurnFractionLabel);
  VarLabel::destroy(AccArchesNCLabel);
  VarLabel::destroy(frictionalWorkLabel);
  VarLabel::destroy(gNumNearParticlesLabel);

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

  VarLabel::destroy(dispNewLabel);
  VarLabel::destroy(dispIncLabel);
  VarLabel::destroy(pAccelerationLabel);
  VarLabel::destroy(dispIncQNorm0);
  VarLabel::destroy(dispIncNormMax);
  VarLabel::destroy(dispIncQNorm);
  VarLabel::destroy(dispIncNorm);
  VarLabel::destroy(pAccelerationLabel_preReloc);
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);

}

void MPMLabel::registerPermanentParticleState(int i,
					      const VarLabel* label,
					      const VarLabel* preReloc_label)
{
  d_particleState[i].push_back(label);
  d_particleState_preReloc[i].push_back(preReloc_label);
}
