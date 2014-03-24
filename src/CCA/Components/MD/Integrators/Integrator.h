/*
 * Integrator.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

namespace Uintah {
  enum MDIntegrator {
    velocityVerlet
  };


  class Integrator {
    public:
      virtual void advanceTime() const = 0;
      virtual std::string getType() const = 0;
    private:
      MDIntegrator type;
  };
}



#endif /* INTEGRATOR_H_ */
