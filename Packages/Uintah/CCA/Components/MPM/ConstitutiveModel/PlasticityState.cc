#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityState.h>
using namespace Uintah;

PlasticityState::PlasticityState()
{
  plasticStrainRate = 0.0;
  plasticStrain = 0.0;
  pressure = 0.0;
  temperature = 0.0;
  density = 0.0;
  initialDensity = 0.0;
  shearModulus = 0.0;
  initialShearModulus = 0.0;
  meltingTemp = 0.0;
  initialMeltTemp = 0.0;
}

PlasticityState::PlasticityState(const PlasticityState& state)
{
  plasticStrainRate = state.plasticStrainRate ;
  plasticStrain = state.plasticStrain ;
  pressure = state.pressure ;
  temperature = state.temperature ;
  density = state.density ;
  initialDensity = state.initialDensity ;
  shearModulus = state.shearModulus ;
  initialShearModulus = state.initialShearModulus ;
  meltingTemp = state.meltingTemp ;
  initialMeltTemp = state.initialMeltTemp ;
}

PlasticityState::~PlasticityState()
{
}

PlasticityState&
PlasticityState::operator=(const PlasticityState& state)
{
  if (this == &state) return *this;
  plasticStrainRate = state.plasticStrainRate ;
  plasticStrain = state.plasticStrain ;
  pressure = state.pressure ;
  temperature = state.temperature ;
  density = state.density ;
  initialDensity = state.initialDensity ;
  shearModulus = state.shearModulus ;
  initialShearModulus = state.initialShearModulus ;
  meltingTemp = state.meltingTemp ;
  initialMeltTemp = state.initialMeltTemp ;
  return *this;
}
