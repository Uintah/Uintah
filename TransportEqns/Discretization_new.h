#ifndef Uintah_Component_Arches_Discretization_new_h
#define Uintah_Component_Arches_Discretization_new_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#define YDIM
#define ZDIM
//==========================================================================

/**
* @class Discretization_new
* @author Jeremy Thornock
* @date Oct 16, 2008
*
* @brief A discretization toolbox.
*       
*
*
*/

namespace Uintah{
  class Discretization_new {

  public:

    Discretization_new();
    ~Discretization_new();

    /** @brief Computes the convection term */
    template <class fT, class oldPhiT> void 
    computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
                constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
                constSFCZVariable<double>& wVel, constCCVariable<double>& den,
                std::string convScheme);

  }; // class Discretization_new

  template<class T> 
  struct FaceData {
    // 0 = e, 1=w, 2=n, 3=s, 4=t, 5=b
    //vector<T> values_[6];
    T p; 
    T e; 
    T w; 
    T n; 
    T s;
    T t;
    T b;
  };

} // namespace Uintah
#endif
