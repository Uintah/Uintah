#ifndef Uintah_MPM_Equation
#define Uintah_MPM_Equation

namespace Uintah {
namespace MPM {

class Equation {
public:
  void             solve();
                   Equation();
  double           mat[4][4];
  double           vec[4];
};

template<class T>
void swap(T& a, T& b);

}} //namespace

#endif

// $Log$
// Revision 1.1  2000/07/05 23:12:30  tan
// Added equation class for least square approximation.
//
