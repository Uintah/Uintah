#ifndef _SPECIFIC_HEAT_MODELFACTORY_H_
#define _SPECIFIC_HEAT_MODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  // Forward declarations
  class SpecificHeatModel;

  /*! \class SpecificHeatModelFactory
   *  \brief Creates instances of Specific Heat Models
   *  \author  Biswajit Banerjee,
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2005 Container Dynamics Group
  */

  class SpecificHeatModelFactory {

  public:

    //! Create a shear modulus model from the input file problem specification.
    static SpecificHeatModel* create(ProblemSpecP& ps);
    static SpecificHeatModel* createCopy(const SpecificHeatModel* yc);
  };
} // End namespace Uintah
      
#endif /* _SPECIFIC_HEAT_MODELFACTORY_H_ */
