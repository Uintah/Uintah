/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
  porosity = 0.0;
  energy = 0.0;
  backStress = Matrix3(0.0);
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
  porosity = state.porosity;
  energy = state.energy;
  backStress = state.backStress;
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
  porosity = state->porosity;
  energy = state->energy;
  backStress = state->backStress;
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
  porosity = state.porosity;
  energy = state.energy;
  backStress = state.backStress;
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
  porosity = state->porosity;
  energy = state->energy;
  backStress = state->backStress;
  return this;
}
