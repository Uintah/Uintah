/*
 * lucretiusAtomMap.h
 *
 *  Created on: Mar 26, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUSATOMMAP_H_
#define LUCRETIUSATOMMAP_H_

#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Exceptions/InvalidValue.h>

#include <CCA/Components/MD/atomMap.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Forcefields/Lucretius/lucretiusAtomData.h>

#include <sstream>

namespace Uintah {

  typedef std::pair<std::string,std::vector<atomData*>*> lucretiusMapPair;
  typedef std::map<std::string,std::vector<atomData*>*> lucretiusMap;
  typedef lucretiusMap::iterator lucretiusMapIterator;
  typedef lucretiusMap::const_iterator constLucretiusMapIterator;

  class lucretiusAtomMap : public atomMap {
    public:
     ~lucretiusAtomMap();
      lucretiusAtomMap(const ProblemSpecP&,
                       const SimulationStateP&,
                       const Forcefield*);

      inline std::vector<atomData*>* operator[](const std::string& searchLabel)
      {
        lucretiusMapIterator labelLocation = findValidAtomList(searchLabel);
        if (labelLocation != atomSet.end())
        {
          return (labelLocation->second);
        }
        else
        {
          return (NULL);
        }
      }

      inline std::vector<atomData*>* getAtomList(const std::string& searchLabel)
      {
        lucretiusMapIterator labelLocation = findValidAtomList(searchLabel);
        if (labelLocation != atomSet.end())
        {
          return (labelLocation->second);
        }
        else
        {
          return (NULL);
        }
      }

      inline size_t getAtomListSize(const std::string& searchLabel) const
      {
        constLucretiusMapIterator labelLocation = findValidAtomList(searchLabel);
        if (labelLocation != atomSet.end())
        {
          return (labelLocation->second->size());
        }
        else
        {
          return (0);
        }
      }

      inline size_t getNumberAtomTypes() const
      {
        return atomSet.size();
      }

      void outputStatistics() const;

    private:
      constLucretiusMapIterator findValidAtomList(const std::string&) const;
      lucretiusMapIterator      findValidAtomList(const std::string&);
      size_t                    addAtomToList(const std::string&, atomData*);

      lucretiusMap atomSet;
  };
}



#endif /* LUCRETIUSATOMMAP_H_ */
