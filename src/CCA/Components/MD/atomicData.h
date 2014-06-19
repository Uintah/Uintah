/*
 * atomicData.h
 *
 *  Created on: Mar 26, 2014
 *      Author: jbhooper
 */

#ifndef ATOMICDATA_H_
#define ATOMICDATA_H_

#include <Core/Geometry/Vector.h>
#include <Core/Disclosure/TypeUtils.h>

#include <vector>

namespace Uintah {

  class atomData {
    public:
      atomData() {};
      virtual ~atomData() {};
      virtual SCIRun::Point getPosition() const = 0;
      virtual SCIRun::Vector getVelocity() const = 0;
      virtual long64 getID() const = 0;
      virtual std::string getLabel() const = 0;
    private:
      // Prevent copying or assignment (slicing)
      atomData& operator=(const atomData&);
      atomData(const atomData&);
  };
}


#endif /* ATOMICDATA_H_ */
