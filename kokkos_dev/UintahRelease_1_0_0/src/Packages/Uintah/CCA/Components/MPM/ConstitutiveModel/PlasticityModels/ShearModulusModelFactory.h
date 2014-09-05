#ifndef _SHEARMODULUSMODELFACTORY_H_
#define _SHEARMODULUSMODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  // Forward declarations
  class ShearModulusModel;

  /*! \class ShearModulusModelFactory
   *  \brief Creates instances of Shear Modulus Models
   *  \author  Biswajit Banerjee,
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2004 Container Dynamics Group
  */

  class ShearModulusModelFactory {

  public:

    //! Create a shear modulus model from the input file problem specification.
    static ShearModulusModel* create(ProblemSpecP& ps);
    static ShearModulusModel* createCopy(const ShearModulusModel* yc);
  };
} // End namespace Uintah
      
#endif /* _SHEARMODULUSMODELFACTORY_H_ */
