/*
 * lucretiusAtomData.h
 *
 *  Created on: Mar 26, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUSATOMDATA_H_
#define LUCRETIUSATOMDATA_H_

#include <CCA/Components/MD/atomicData.h>
#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>

#include <string>

namespace Uintah {

  class lucretiusAtomData : public atomData {
    public:
      lucretiusAtomData(const SCIRun::Point,
                                 const SCIRun::Vector,
                                 size_t,
                                 const std::string&,
                                 size_t,
                                 const forcefieldType _ff=Lucretius);
      lucretiusAtomData(double, double, double,
                                 double, double, double,
                                 size_t,
                                 const std::string&,
                                 size_t,
                                 const forcefieldType _ff=Lucretius);
      virtual ~lucretiusAtomData() {};
      virtual SCIRun::Point getPosition() const {
        return d_Position;
      }
      virtual SCIRun::Vector getVelocity() const {
        return d_Velocity;
      }
      virtual long64 getID() const {
        return d_ParticleID;
      }
      virtual std::string getLabel() const {
        return d_Label;
      }

    private:
      SCIRun::Point d_Position;
      SCIRun::Vector d_Velocity;
      size_t d_ParticleID;
      std::string d_Label;
      const forcefieldType d_forcefield;

      // Prevent copy and assignment
      lucretiusAtomData& operator=(const lucretiusAtomData&);
      lucretiusAtomData(const lucretiusAtomData&);
  };
}



#endif /* LUCRETIUSATOMDATA_H_ */
