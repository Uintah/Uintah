/*
 * IntegratorFactory.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef INTEGRATORFACTORY_H_
#define INTEGRATORFACTORY_H_

namespace Uintah {

  class ProcessorGroup;
  class MDSystem;

  class IntegratorFactory {
    public:
      static Integrator* create(const ProblemSpecP& ps,
                                      MDSystem*     system,
                                const VarLabel*     del_time);
  };
}  // namespace Uintah

#endif /* INTEGRATORFACTORY_H_ */
