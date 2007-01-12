#include "PlasticityState.h"
using namespace Uintah;

PlasticityState::PlasticityState()
{
  yieldStress = 0.0;
  strainRate = 0.0;
  plasticStrainRate = 0.0;
  plasticStrain = 0.0;
  pressure = 0.0;
  temperature = 0.0;
  initialTemperature = 0.0;
  density = 0.0;
  initialDensity = 0.0;
  volume = 0.0;
  initialVolume = 0.0;
  bulkModulus = 0.0;
  initialBulkModulus = 0.0;
  shearModulus = 0.0;
  initialShearModulus = 0.0;
  meltingTemp = 0.0;
  initialMeltTemp = 0.0;
  specificHeat = 0.0;
}

PlasticityState::PlasticityState(const PlasticityState& state)
{
  yieldStress = state.yieldStress ;
  strainRate = state.strainRate;
  plasticStrainRate = state.plasticStrainRate ;
  plasticStrain = state.plasticStrain ;
  pressure = state.pressure ;
  temperature = state.temperature ;
  initialTemperature = state.initialTemperature ;
  density = state.density ;
  initialDensity = state.initialDensity ;
  volume = state.volume ;
  initialVolume = state.initialVolume ;
  bulkModulus = state.bulkModulus ;
  initialBulkModulus = state.initialBulkModulus ;
  shearModulus = state.shearModulus ;
  initialShearModulus = state.initialShearModulus ;
  meltingTemp = state.meltingTemp ;
  initialMeltTemp = state.initialMeltTemp ;
  specificHeat = state.specificHeat;
}

PlasticityState::PlasticityState(const PlasticityState* state)
{
  yieldStress = state->yieldStress ;
  strainRate = state->strainRate;
  plasticStrainRate = state->plasticStrainRate ;
  plasticStrain = state->plasticStrain ;
  pressure = state->pressure ;
  temperature = state->temperature ;
  initialTemperature = state->initialTemperature ;
  density = state->density ;
  initialDensity = state->initialDensity ;
  volume = state->volume ;
  initialVolume = state->initialVolume ;
  bulkModulus = state->bulkModulus ;
  initialBulkModulus = state->initialBulkModulus ;
  shearModulus = state->shearModulus ;
  initialShearModulus = state->initialShearModulus ;
  meltingTemp = state->meltingTemp ;
  initialMeltTemp = state->initialMeltTemp ;
  specificHeat = state->specificHeat;
}

PlasticityState::~PlasticityState()
{
}

PlasticityState&
PlasticityState::operator=(const PlasticityState& state)
{
  if (this == &state) return *this;
  yieldStress = state.yieldStress ;
  strainRate = state.strainRate;
  plasticStrainRate = state.plasticStrainRate ;
  plasticStrain = state.plasticStrain ;
  pressure = state.pressure ;
  temperature = state.temperature ;
  initialTemperature = state.initialTemperature ;
  density = state.density ;
  initialDensity = state.initialDensity ;
  volume = state.volume ;
  initialVolume = state.initialVolume ;
  bulkModulus = state.bulkModulus ;
  initialBulkModulus = state.initialBulkModulus ;
  shearModulus = state.shearModulus ;
  initialShearModulus = state.initialShearModulus ;
  meltingTemp = state.meltingTemp ;
  initialMeltTemp = state.initialMeltTemp ;
  specificHeat = state.specificHeat;
  return *this;
}

PlasticityState*
PlasticityState::operator=(const PlasticityState* state)
{
  if (this == state) return this;
  yieldStress = state->yieldStress ;
  strainRate = state->strainRate;
  plasticStrainRate = state->plasticStrainRate ;
  plasticStrain = state->plasticStrain ;
  pressure = state->pressure ;
  temperature = state->temperature ;
  initialTemperature = state->initialTemperature ;
  density = state->density ;
  initialDensity = state->initialDensity ;
  volume = state->volume ;
  initialVolume = state->initialVolume ;
  bulkModulus = state->bulkModulus ;
  initialBulkModulus = state->initialBulkModulus ;
  shearModulus = state->shearModulus ;
  initialShearModulus = state->initialShearModulus ;
  meltingTemp = state->meltingTemp ;
  initialMeltTemp = state->initialMeltTemp ;
  specificHeat = state->specificHeat;
  return this;
}
