#ifndef UINTAH_MPM_EQUATION
#define UINTAH_MPM_EQUATION

namespace Uintah {

class Equation {
public:
  void             solve();
                   Equation();
  double           mat[4][4];
  double           vec[4];
};

template<class T>
void swap(T& a, T& b);

} // End namespace Uintah

#endif
