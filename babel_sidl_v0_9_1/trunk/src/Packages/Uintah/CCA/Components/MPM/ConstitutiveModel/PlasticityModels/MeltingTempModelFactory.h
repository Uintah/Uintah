#ifndef _MELTING_TEMP_MODELFACTORY_H_
#define _MELTING_TEMP_MODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  // Forward declarations
  class MeltingTempModel;

  /*! \class MeltingTempModelFactory
   *  \brief Creates instances of Melting Temp Models
   *  \author  Biswajit Banerjee,
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2004 Container Dynamics Group
  */

  class MeltingTempModelFactory {

  public:

    //! Create a melting temp model from the input file problem specification.
    static MeltingTempModel* create(ProblemSpecP& ps);
    static MeltingTempModel* createCopy(const MeltingTempModel* yc);
  };
} // End namespace Uintah
      
#endif /* _MELTING_TEMP_MODELFACTORY_H_ */
