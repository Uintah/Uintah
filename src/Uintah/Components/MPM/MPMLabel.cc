#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Grid/VarTypes.h>

using namespace Uintah;
using namespace Uintah::MPM;

MPMLabel::MPMLabel()
{
  pDeformationMeasureLabel = new VarLabel("p.deformationMeasure",
			    ParticleVariable<Matrix3>::getTypeDescription());
  
  pStressLabel = new VarLabel( "p.stress",
			       ParticleVariable<Matrix3>::getTypeDescription() );
  
  pVolumeLabel = new VarLabel( "p.volume",
			       ParticleVariable<double>::getTypeDescription());
  
  pMassLabel = new VarLabel( "p.mass",
			     ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel = new VarLabel( "p.velocity", 
				 ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel = new VarLabel( "p.externalforce",
				      ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel = new VarLabel( "p.x", ParticleVariable<Point>::getTypeDescription(),
			  VarLabel::PositionVariable);
  
  pTemperatureLabel = new VarLabel( "p.temperature",
				    ParticleVariable<double>::getTypeDescription() );

  
  pTemperatureGradientLabel = new VarLabel( "p.temperatureGradient",
					    ParticleVariable<Vector>::getTypeDescription() );

  pTemperatureRateLabel  = new VarLabel( "p.temperatureRate",
				    ParticleVariable<double>::getTypeDescription() );
  
  //tan:
  //  pSurfaceNormalLabel is used to define the surface normal of a boundary particle.
  //  For the interior particle, the p.surfaceNormal vector is set to (0,0,0)
  //  in this way we can distinguish boundary particles to interior particles
  //
  pSurfaceNormalLabel = new VarLabel( "p.surfaceNormal",
				      ParticleVariable<Vector>::getTypeDescription() );
  
  gAccelerationLabel = new VarLabel( "g.acceleration",
				     NCVariable<Vector>::getTypeDescription() );
  
  gMomExedAccelerationLabel = new VarLabel( "g.momexedacceleration",
					    NCVariable<Vector>::getTypeDescription() );
  
  gMassLabel = new VarLabel( "g.mass",
			     NCVariable<double>::getTypeDescription() );
  
  gVelocityLabel = new VarLabel( "g.velocity",
				 NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityLabel = new VarLabel( "g.momexedvelocity",
					NCVariable<Vector>::getTypeDescription() );
  
  gExternalForceLabel = new VarLabel( "g.externalforce",
				      NCVariable<Vector>::getTypeDescription() );
  
  gInternalForceLabel = new VarLabel( "g.internalforce",
				      NCVariable<Vector>::getTypeDescription() );
  
  gVelocityStarLabel = new VarLabel( "g.velocity_star",
				     NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityStarLabel = new VarLabel( "g.momexedvelocity_star",
					    NCVariable<Vector>::getTypeDescription() );
  
  gSelfContactLabel = new VarLabel( "g.selfContact",
				    NCVariable<bool>::getTypeDescription() );
  
  cSelfContactLabel = new VarLabel( "c.selfContact",
				    CCVariable<bool>::getTypeDescription() );
  
  cSurfaceNormalLabel = new VarLabel( "c.surfaceNormalLabel",
				      CCVariable<Vector>::getTypeDescription() );
  
  gTemperatureRateLabel = new VarLabel("g.temperatureRate",
			       NCVariable<double>::getTypeDescription());

  gInternalHeatRateLabel = new VarLabel("g.internalTemperatureRate",
				NCVariable<double>::getTypeDescription());

  gExternalHeatRateLabel = new VarLabel("g.externalTemperatureRate",
				NCVariable<double>::getTypeDescription());
  
  // I'm not sure about this one:
  delTLabel = 
    new VarLabel( "delT", delt_vartype::getTypeDescription() );
} 

const MPMLabel* MPMLabel::getLabels()
{
  static MPMLabel* instance=0;
  if(!instance)
    instance=new MPMLabel();
  return instance;
}


