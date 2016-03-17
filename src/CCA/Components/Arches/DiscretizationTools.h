#ifndef Uintah_Component_Arches_DISCRETIZATIONTOOLS_h
#define Uintah_Component_Arches_DISCRETIZATIONTOOLS_h

#include <Core/Exceptions/InvalidValue.h>

/** @class DiscretizationTools
    @author J. Thornock
    @date Dec, 2015

    @brief Provides some basic, commonly used/shared functionality for differencing
    grid variables.

**/

namespace Uintah{

  enum DIR {XDIR, YDIR, ZDIR};
  enum LIMITER {CENTRAL, UPWIND, SUPERBEE, ROE, VANLEER};

// IDIR, JDIR, KDIR are relative to the CC variable.
#define IDIR \
      IntVector c(i,j,k); \
      IntVector cm(i-1,j,k); \
      IntVector cp(i+1,j,k); \
      IntVector cmm(i-2,j,k); \
      IntVector cpp(i+2,j,k);

#define JDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j-1,k); \
      IntVector cp(i,j+1,k); \
      IntVector cmm(i,j-2,k); \
      IntVector cpp(i,j+2,k);

#define KDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j,k-1); \
      IntVector cp(i,j,k+1); \
      IntVector cmm(i,j,k-2); \
      IntVector cpp(i,j,k+2);

// Offsets for the staggered cells. We have ABDIR where A=staggered cell direction and
// I is the compass direction for the differencing.
#define XIDIR \
      IntVector c(i,j,k); \
      IntVector cm(i-1,j,k); \
      IntVector cp(i+1,j,k); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i-1,j,k); \
      IntVector cu_e(i,j,k); \
      IntVector cu_e2(i+1,j,k);

#define XJDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j-1,k); \
      IntVector cp(i,j+1,k); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i-1,j,k); \
      IntVector cu_e(i,j+1,k); \
      IntVector cu_e2(i-1,j+1,k);

#define XKDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j,k-1); \
      IntVector cp(i,j,k+1); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i-1,j,k); \
      IntVector cu_e(i,j,k+1); \
      IntVector cu_e2(i-1,j,k+1);

#define YIDIR \
      IntVector c(i,j,k); \
      IntVector cm(i-1,j,k); \
      IntVector cp(i+1,j,k); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i,j-1,k); \
      IntVector cu_e(i+1,j,k); \
      IntVector cu_e2(i+1,j-1,k);

#define YJDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j-1,k); \
      IntVector cp(i,j+1,k); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i,j-1,k); \
      IntVector cu_e(i,j,k); \
      IntVector cu_e2(i,j+1,k);

#define YKDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j,k-1); \
      IntVector cp(i,j,k+1); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i,j-1,k); \
      IntVector cu_e(i,j,k+1); \
      IntVector cu_e2(i,j-1,k+1);

#define ZIDIR \
      IntVector c(i,j,k); \
      IntVector cm(i-1,j,k); \
      IntVector cp(i+1,j,k); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i,j,k-1); \
      IntVector cu_e(i+1,j,k); \
      IntVector cu_e2(i+1,j,k-1);

#define ZJDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j-1,k); \
      IntVector cp(i,j+1,k); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i,j,k-1); \
      IntVector cu_e(i,j+1,k); \
      IntVector cu_e2(i,j+1,k-1);

#define ZKDIR \
      IntVector c(i,j,k); \
      IntVector cm(i,j,k-1); \
      IntVector cp(i,j,k+1); \
      IntVector cu_w(i,j,k); \
      IntVector cu_w2(i,j,k-1); \
      IntVector cu_e(i,j,k); \
      IntVector cu_e2(i,j,k+1);

  /**
      @struct VariableHelper
      @details Useful for reconciling the const and non-const types for a specific
      template parameter. Also useful for deducing face types.
  **/
  template <typename T>
  struct VariableHelper{
  };

  //Helper specialization:
  template <>
  struct VariableHelper<CCVariable<double> >{
    typedef constCCVariable<double> ConstType;
    typedef CCVariable<double> Type;
    typedef SFCXVariable<double> XFaceType;
    typedef SFCYVariable<double> YFaceType;
    typedef SFCZVariable<double> ZFaceType;
  };

  template <>
  struct VariableHelper<SFCXVariable<double> >{
    typedef constSFCXVariable<double> ConstType;
    typedef SFCXVariable<double> Type;
    typedef CCVariable<double> XFaceType;
    typedef SFCYVariable<double> YFaceType;
    typedef SFCZVariable<double> ZFaceType;
  };

  template <>
  struct VariableHelper<SFCYVariable<double> >{
    typedef constSFCYVariable<double> ConstType;
    typedef SFCYVariable<double> Type;
    typedef SFCXVariable<double> XFaceType;
    typedef CCVariable<double> YFaceType;
    typedef SFCZVariable<double> ZFaceType;
  };

  template <>
  struct VariableHelper<SFCZVariable<double> >{
    typedef constSFCZVariable<double> ConstType;
    typedef SFCZVariable<double> Type;
    typedef SFCXVariable<double> XFaceType;
    typedef SFCYVariable<double> YFaceType;
    typedef CCVariable<double> ZFaceType;
  };

  //------------------------------------------------------------------------------------------------
  // These stucts below need a good home:

  /**
      @struct VariableConstantInitializeFunctor
      @details Initialize a grid variable to a constant
      This doesn't have a good home at the moment so it resides here.
  **/
  template <typename T>
  struct VariableConstantInitializeFunctor{

    T& var;
    double value;

    VariableConstantInitializeFunctor( T& var, double value )
      : var(var), value(value){}

    void operator()(int i, int j, int k) const{

      const IntVector c(i,j,k);

      var[c] = value;

    }
  };

  /**
      @struct VariableStepInitializeFunctor
      @details Initialize a grid variable to a step constant.
      This doesn't have a good home at the moment so it resides here.
  **/
  template <typename T>
  struct VariableStepInitializeFunctor{

    T& var;
    constCCVariable<double>& gridX;
    double value;

    VariableStepInitializeFunctor( T& var, constCCVariable<double>& gridX, double value )
      : var(var), gridX(gridX), value(value){}

    void operator()(int i, int j, int k) const{

      const IntVector c(i,j,k);
      double start = 0.5;

      double value_assign = (gridX[c] > start) ? 0.0 : value;

      var[c] = value_assign;

    }
  };

  /**
      @struct ComputeDiffusion
      @details Compute the diffusion term for a scalar.
      This doesn't have a good home at the moment so it resides here.
      @TODO Currently only works for CCVariables. Generalize it?
      @TODO Fix it to work with boundaries.
  **/
  template <typename T, typename FT>
  struct ComputeDiffusion{

    //for now assuming that this will only be used for CC variables:
    typedef typename VariableHelper<T>::ConstType ConstPT;
    typedef typename VariableHelper<T>::Type PT;

    ConstPT& phi;
    constCCVariable<double>& gamma; //!!!!!!!
    FT& af;
    PT& rhs;
    IntVector dir;
    double A;
    double dx;

    ComputeDiffusion( ConstPT& phi, constCCVariable<double>& gamma, PT& rhs, FT& af,
      IntVector dir, double A, double dx ) :
      phi(phi), gamma(gamma), rhs(rhs), af(af), dir(dir), A(A), dx(dx) { }

    void operator()(int i, int j, int k) const {

      IntVector c(i,j,k);
      IntVector cm(i-dir[0],j-dir[1],k-dir[2]);
      IntVector cp(i+dir[0],j+dir[1],k+dir[2]);

      double face_gamma_e = (gamma[c] + gamma[cp])/ 2.0 * af[cp];
      double face_gamma_w = (gamma[c] + gamma[cm])/ 2.0 * af[c];

      double grad_e = (phi[cp]-phi[c])/dx;
      double grad_w = (phi[c]-phi[cm])/dx;

      rhs[c] += A * ( face_gamma_e * grad_e - face_gamma_w * grad_w );

    }
  };

  //discretization
  class DiscretizationTools{
  public:

  private:

  };
}
#endif
