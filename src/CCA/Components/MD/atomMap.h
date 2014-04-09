/*
 * atomMap.h
 *
 *  Created on: Mar 26, 2014
 *      Author: jbhooper
 */

#ifndef ATOMMAP_H_
#define ATOMMAP_H_

#include <CCA/Components/MD/atomicData.h>

namespace Uintah {

  class atomMap {
    public:
      atomMap() {}
      virtual ~atomMap() {}
      virtual std::vector<atomData*>* operator[](const std::string&) = 0;
      virtual std::vector<atomData*>* getAtomList(const std::string&) = 0;
      virtual size_t getAtomListSize(const std::string&) = 0;
      virtual size_t addAtomToList(const std::string&, atomData*) = 0;
      virtual size_t getNumberAtomTypes() const = 0;
    private:
  };

}

#endif /* ATOMMAP_H_ */
