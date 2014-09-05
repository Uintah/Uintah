/*
 * Integrator.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

namespace Uintah {
  enum interactionModel { Deterministic, Stochastic, Mixed };

  enum MDIntegrator {
    velocityVerlet
  };


  class Integrator {
    public:
      Integrator() {};
      virtual void advanceTime() const = 0;
      virtual std::string getType() const = 0;
      virtual ~Integrator() {};
    private:
      MDIntegrator type;
  };
}



#endif /* INTEGRATOR_H_ */
