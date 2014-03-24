/*
 * nonbondedPotentialMapKey.h
 *
 *  Created on: Mar 19, 2014
 *      Author: jbhooper
 */

#ifndef NONBONDEDPOTENTIALMAPKEY_H_
#define NONBONDEDPOTENTIALMAPKEY_H_

#include <map>

#include<CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>

namespace Uintah {

  class nonbondedTwoBodyKey {
    public:
      nonbondedTwoBodyKey(std::string _1,std::string _2):firstLabel(_1),secondLabel(_2) { }
      ~nonbondedTwoBodyKey() { }

      inline bool operator < (const nonbondedTwoBodyKey& rhs) const {
        if (firstLabel == rhs.firstLabel) {
          return (secondLabel < rhs.secondLabel);
        }
        return (firstLabel < rhs.firstLabel);
      }
    private:
      std::string firstLabel;
      std::string secondLabel;
  };

  typedef std::pair<nonbondedTwoBodyKey, NonbondedTwoBodyPotential*> twoBodyPotentialMapPair;
  typedef std::map<nonbondedTwoBodyKey,NonbondedTwoBodyPotential*> nonbondedTwoBodyMapType;

}



#endif /* NONBONDEDPOTENTIALMAPKEY_H_ */
