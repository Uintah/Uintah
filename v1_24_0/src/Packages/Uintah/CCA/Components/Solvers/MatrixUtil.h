
#ifndef Packages_Uintah_CCA_Components_Solvers_MatrixUtil_h
#define Packages_Uintah_CCA_Components_Solvers_MatrixUtil_h

#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>

namespace Uintah {
  class SFCXTypes {
  public:
    typedef constSFCXVariable<Stencil7> matrix_type;
    typedef constSFCXVariable<double> const_type;
    typedef SFCXVariable<double> sol_type;
  };

  class SFCYTypes {
  public:
    typedef constSFCYVariable<Stencil7> matrix_type;
    typedef constSFCYVariable<double> const_type;
    typedef SFCYVariable<double> sol_type;
  };

  class SFCZTypes {
  public:
    typedef constSFCZVariable<Stencil7> matrix_type;
    typedef constSFCZVariable<double> const_type;
    typedef SFCZVariable<double> sol_type;
  };

  class CCTypes {
  public:
    typedef constCCVariable<Stencil7> matrix_type;
    typedef constCCVariable<double> const_type;
    typedef CCVariable<double> sol_type;
  };
  
  class NCTypes {
  public:
    typedef constNCVariable<Stencil7> matrix_type;
    typedef constNCVariable<double> const_type;
    typedef NCVariable<double> sol_type;
  };
}

#endif

