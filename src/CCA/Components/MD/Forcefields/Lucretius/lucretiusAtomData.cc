/*
 * lucretiusAtomData.cc
 *
 *  Created on: Mar 26, 2014
 *      Author: jbhooper
 */

#include <Core/Geometry/Vector.h>

#include <CCA/Components/MD/Forcefields/Lucretius/lucretiusAtomData.h>
#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>
#include <Core/Exceptions/InvalidValue.h>

#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

using namespace Uintah;

lucretiusAtomData::lucretiusAtomData(double _X1, double _X2, double _X3,
                                     double _V1, double _V2, double _V3,
                                     size_t _ID,
                                     const std::string& _nbLabel,
                                     size_t _chargeIndex,
                                     const forcefieldType _ff):
                                         d_Position(SCIRun::Vector(_X1,_X2,_X3)),
                                         d_Velocity(SCIRun::Vector(_V1,_V2,_V3)),
                                         d_ParticleID(_ID),
                                         d_forcefield(_ff) {

  if (_nbLabel.size() > 3) {
    throw InvalidValue("ERROR:  Lucretius label should be 3 characters or less.", __FILE__, __LINE__);
  }
  std::stringstream lucretiusLabel;
  std::string tempLabel(3,' ');
  for (size_t index=0; index < _nbLabel.size(); ++index) {
    tempLabel[index]=_nbLabel[index];
  }
  lucretiusLabel << tempLabel << std::ios::left << _chargeIndex;
  d_Label = lucretiusLabel.str();

}

lucretiusAtomData::lucretiusAtomData(const SCIRun::Point _X,
                                     const SCIRun::Vector _V,
                                     size_t  _ID,
                                     const std::string& _nbLabel,
                                     size_t _chargeIndex,
                                     const forcefieldType _ff):
                                         d_Position(_X),
                                         d_Velocity(_V),
                                         d_ParticleID(_ID),
                                         d_forcefield(_ff) {

  if (_nbLabel.size() > 3) {
    throw InvalidValue("ERROR:  Lucretius label should be 3 characters or less.", __FILE__, __LINE__);
  }
  std::stringstream lucretiusLabel;
  std::string tempLabel(3,' ');
  for (size_t index=0; index < _nbLabel.size(); ++index) {  // Copy over the nonbonded label while leaving padding spaces
    tempLabel[index]=_nbLabel[index];
  }
  lucretiusLabel << tempLabel << std::ios::left << _chargeIndex;
  d_Label = lucretiusLabel.str();

}

