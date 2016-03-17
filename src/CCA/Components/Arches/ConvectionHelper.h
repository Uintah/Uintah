#ifndef Uintah_Component_Arches_CONVECTIONHELPER_h
#define Uintah_Component_Arches_CONVECTIONHELPER_h

#include <CCA/Components/Arches/DiscretizationTools.h>

/** @class ConvectionHelper
    @author J. Thornock
    @date Dec, 2015

    @brief This class provides support for including the convection operator
    for building transport of various grid variables.
**/

/* DETAILS
   There are a three things going on here:
   1) Definition of a templated functor, ComputeConvection, which computes the 1D
   implementation of a convection operator.
   2) Definition of a set of macros for the actual convection operator.
   3) Template specialization of 1).
   To add a new convection operator, one must:
   a) Define the appropriate macros for the operator
   b) specialize ComputeConvection for the supported grid-variabe types. Note that
   the default template will be called for non-supported types and throw and error
   in the actual operator() if this is ever hit during run-time.
   c) potentially define a convenience macro (eg, UPWIND_CONVECTION) to run through
   the three different directions is applicable.
*/

namespace Uintah{

// Assumes a face variable and a face velocity.
#define CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2) \
      double phi_e = ( phi[c] + phi[cp] )/ 2. * af[cp]; \
      double phi_w = ( phi[c] + phi[cm] )/ 2. * af[c]; \
      double u_e = ( u[cu_e] + u[cu_e2] ) / 2.; \
      double u_w = ( u[cu_w] + u[cu_w2] ) / 2.; \
      rhs[c] += -A * ( phi_e * u_e - phi_w * u_w );

#define SUPERBEEMACRO(r) \
      psi = ( r < huge ) ? std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) ) : 2.0; \
      psi = std::max( 0.0, psi ) \

#define ROEMACRO(r) \
      psi = ( r < huge ) ? std::min(r, 0.0) : 2.0; \
      psi = std::max(0.0, psi);

#define CENTRALMACRO(r) \
      psi = 1.0;

#define UPWINDMACRO(r) \
      psi = 0.0;

#define VANLEERMACRO(r) \
      psi = ( r < huge ) ? ( r + std::abs(r) ) / ( 1. + std::abs(r) ) : 0.;

// Generic flux limiter. Assumes that the velocities are at the face so no interplation
// is done to u. Once the flux is constructed, it sums it into the RHS.
#define FLUXLIM(c,cm,cp,cmm,cpp, LIMTYPE) \
      double psi; \
      double psi_up; \
      double psi_dn; \
      double Sup_up = phi[cm]; \
      double Sdn_up = phi[c]; \
      double r = (phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cm]; \
      double saver1= r; \
      LIMTYPE(r); \
      psi_up = psi; \
      double savepsi1 = psi; \
      double Sup_dn = phi[c]; \
      double Sdn_dn = phi[cm]; \
      r = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cp]; \
      double saver2= r; \
      LIMTYPE(r); \
      double savepsi2 = psi; \
      psi_dn = psi; \
      double face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up ); \
      double face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn ); \
      double face_value_m = ( u[c] > 0.0 ) ? face_up : face_dn; \
      Sup_up = phi[c]; \
      Sdn_up = phi[cp]; \
      r = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] ); \
      r *= af[c] * af[cp];  \
      double saver3= r; \
      LIMTYPE(r); \
      double savepsi3 = psi; \
      psi_up = psi; \
      Sup_dn = phi[cp]; \
      Sdn_dn = phi[c];  \
      r =  ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] ); \
      r *= af[cp]*af[cpp]; \
      double saver4= r; \
      LIMTYPE(r); \
      double savepsi4=psi; \
      psi_dn = psi; \
      face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up ); \
      face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn ); \
      double face_value_p = ( u[cp] > 0.0 ) ? face_up : face_dn; \
      rhs[c] += -A * ( face_value_p * u[cp] - face_value_m * u[c] );

// Same as FLUXLIM but performs linearly interpolates in the minus direction to be
// consistent with BCs.
#define FLUXLIM_MINUSBC(c,cm,cp,cmm,cpp, LIMTYPE) \
      double psi; \
      double psi_up; \
      double psi_dn; \
      double Sup_up = phi[cm]; \
      double Sdn_up = phi[c]; \
      double face_value_m = 0.5 * (phi[cm] + phi[c]); \
      Sup_up = phi[c]; \
      Sdn_up = phi[cp]; \
      double r = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] ); \
      r *= af[c] * af[cp];  \
      double saver3= r; \
      LIMTYPE(r); \
      double savepsi3 = psi; \
      psi_up = psi; \
      Sup_dn = phi[cp]; \
      Sdn_dn = phi[c];  \
      r =  ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] ); \
      r *= af[cp]*af[cpp]; \
      double saver4= r; \
      LIMTYPE(r); \
      double savepsi4=psi; \
      psi_dn = psi; \
      face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up ); \
      face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn ); \
      double face_value_p = ( u[cp] > 0.0 ) ? face_up : face_dn; \
      rhs[c] += -A * ( face_value_p * u[cp] - face_value_m * u[c] );

// Same as FLUXLIM but performs linearly interpolates in the plus direction to be
// consistent with BCs.
#define FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, LIMTYPE) \
      double psi; \
      double psi_up; \
      double psi_dn; \
      double Sup_up = phi[cm]; \
      double Sdn_up = phi[c]; \
      double r = (phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cm]; \
      double saver1= r; \
      LIMTYPE(r); \
      psi_up = psi; \
      double savepsi1 = psi; \
      double Sup_dn = phi[c]; \
      double Sdn_dn = phi[cm]; \
      r = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cp]; \
      double saver2= r; \
      LIMTYPE(r); \
      double savepsi2 = psi; \
      psi_dn = psi; \
      double face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up ); \
      double face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn ); \
      double face_value_m = ( u[c] > 0.0 ) ? face_up : face_dn; \
      double face_value_p = 0.5 * ( phi[c] + phi[cp] );  \
      rhs[c] += -A * ( face_value_p * u[cp] - face_value_m * u[c] );

// Computes the psi funcion from the LIMTYPE for the plus and minus faces.
#define GETPSI(c,cm,cp,cmm,cpp, LIMTYPE) \
      double psi; \
      double psi_up; \
      double psi_dn; \
      double r = (phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cm]; \
      LIMTYPE(r); \
      psi_up = psi; \
      r = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cp]; \
      LIMTYPE(r); \
      psi_dn = psi; \
      double psi_save_m = ( u[c] > 0.0 ) ? psi_up : psi_dn; \
      r = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] ); \
      r *= af[c] * af[cp];  \
      LIMTYPE(r); \
      psi_up = psi; \
      r =  ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] ); \
      r *= af[cp]*af[cpp]; \
      LIMTYPE(r); \
      psi_dn = psi; \
      double psi_p = ( u[cp] > 0.0 ) ? psi_up : psi_dn;

// Computes the plus face psi assuming a BC will be applied to the minus face.
#define GETPSI_MINUSBC(c,cm,cp,cmm,cpp, LIMTYPE) \
      double psi; \
      double psi_up; \
      double psi_dn; \
      double psi_m = 1.0; \
      double r = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] ); \
      r *= af[c] * af[cp];  \
      LIMTYPE(r); \
      psi_up = psi; \
      r =  ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] ); \
      r *= af[cp]*af[cpp]; \
      LIMTYPE(r); \
      psi_dn = psi; \
      double psi_p = ( u[cp] > 0.0 ) ? psi_up : psi_dn;

// Computes the minus face psi assuming a BC will be applied to the minus face.
#define GETPSI_PLUSBC(c,cm,cp,cmm,cpp, LIMTYPE) \
      double psi; \
      double psi_up; \
      double psi_dn; \
      double r = (phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cm]; \
      LIMTYPE(r); \
      psi_up = psi;\
      r = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] ); \
      r *= af[c] * af[cp]; \
      LIMTYPE(r); \
      psi_dn = psi; \
      double psi_m = ( u[c] > 0.0 ) ? psi_up : psi_dn; \
      double psi_p = 1.;

#define UPWIND_CONVECTION \
        ComputeConvection<T, constSFCXVariable<double>, XDIR, UPWIND > x_conv( phi, rhs, u, afx, Ayz); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), x_conv ); \
        ComputeConvection<T, constSFCYVariable<double>, YDIR, UPWIND > y_conv( phi, rhs, v, afy, Azx); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), y_conv ); \
        ComputeConvection<T, constSFCZVariable<double>, ZDIR, UPWIND > z_conv( phi, rhs, w, afz, Axy); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), z_conv );

#define SUPERBEE_CONVECTION \
        ComputeConvection<T, constSFCXVariable<double>, XDIR, SUPERBEE > x_conv( phi, rhs, u, afx, Ayz); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), x_conv ); \
        ComputeConvection<T, constSFCYVariable<double>, YDIR, SUPERBEE > y_conv( phi, rhs, v, afy, Azx); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), y_conv ); \
        ComputeConvection<T, constSFCZVariable<double>, ZDIR, SUPERBEE > z_conv( phi, rhs, w, afz, Axy); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), z_conv );

#define VANLEER_CONVECTION \
        ComputeConvection<T, constSFCXVariable<double>, XDIR, VANLEER > x_conv( phi, rhs, u, afx, Ayz); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), x_conv ); \
        ComputeConvection<T, constSFCYVariable<double>, YDIR, VANLEER > y_conv( phi, rhs, v, afy, Azx); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), y_conv ); \
        ComputeConvection<T, constSFCZVariable<double>, ZDIR, VANLEER > z_conv( phi, rhs, w, afz, Axy); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), z_conv );

#define CENTRAL_CONVECTION \
        ComputeConvection<T, constSFCXVariable<double>, XDIR, CENTRAL > x_conv( phi, rhs, u, afx, Ayz); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), x_conv ); \
        ComputeConvection<T, constSFCYVariable<double>, YDIR, CENTRAL > y_conv( phi, rhs, v, afy, Azx); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), y_conv ); \
        ComputeConvection<T, constSFCZVariable<double>, ZDIR, CENTRAL > z_conv( phi, rhs, w, afz, Axy); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), z_conv );

#define ROE_CONVECTION \
        ComputeConvection<T, constSFCXVariable<double>, XDIR, ROE > x_conv( phi, rhs, u, afx, Ayz); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), x_conv ); \
        ComputeConvection<T, constSFCYVariable<double>, YDIR, ROE > y_conv( phi, rhs, v, afy, Azx); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), y_conv ); \
        ComputeConvection<T, constSFCZVariable<double>, ZDIR, ROE > z_conv( phi, rhs, w, afz, Axy); \
        Kokkos::parallel_for( Kokkos::Range3Policy<int>(l[0],l[1],l[2], h[0],h[1],h[2]), z_conv );

  /**
      @struct TagMinusBC
      @brief A struct used to specialize the ComputeConection operator(). Informs the
      function that a BC is present on the minus side.
  **/
  struct TagMinusBC{};
  /**
      @struct TagPlusBC
      @brief A struct used to specialize the ComputeConection operator(). Informs the
      function that a BC is present on the plus side.
  **/
  struct TagPlusBC{};

  /**
      @struct ComputeConvection
      @brief Generic, templated interface for computing convection with this functor. If
      actually used, this function should throw an error since the specialized versions
      are the functors actually doing the work.
  **/
  template<typename PT, typename UT, int MYDIR, int MYLIMITER>
  struct ComputeConvection{
    typedef typename VariableHelper<PT >::ConstType ConstPT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {
      throw InvalidValue(
        "Error: No implementation of this method in DiscretizationTools.h",
        __FILE__, __LINE__);
    }
    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {
      throw InvalidValue(
        "Error: No implementation of this method in DiscretizationTools.h",
        __FILE__, __LINE__);
    }
    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {
      throw InvalidValue(
        "Error: No implementation of this method in DiscretizationTools.h",
        __FILE__, __LINE__);
    }
  };

  //-------------------------------------
  //------- Specialized functions -------
  //-------------------------------------

  /*
          .d8888. db    db d8888b. d88888b d8888b. d8888b. d88888b d88888b
          88'  YP 88    88 88  `8D 88'     88  `8D 88  `8D 88'     88'
          `8bo.   88    88 88oodD' 88ooooo 88oobY' 88oooY' 88ooooo 88ooooo
            `Y8b. 88    88 88~~~   88~~~~~ 88`8b   88~~~b. 88~~~~~ 88~~~~~
          db   8D 88b  d88 88      88.     88 `88. 88   8D 88.     88.
          `8888Y' ~Y8888P' 88      Y88888P 88   YD Y8888P' Y88888P Y88888P
  */

  // --x-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, XDIR, SUPERBEE>{
  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      IDIR;

      FLUXLIM(c,cm,cp,cmm,cpp, SUPERBEEMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, SUPERBEEMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, SUPERBEEMACRO);

    }
  };

  // --y-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, YDIR, SUPERBEE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      JDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,SUPERBEEMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, SUPERBEEMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, SUPERBEEMACRO);

    }
  };

  // --z-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, ZDIR, SUPERBEE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      KDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,SUPERBEEMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, SUPERBEEMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, SUPERBEEMACRO);

    }
  };

  /*
    d8888b.  .d88b.  d88888b      .88b  d88. d888888b d8b   db .88b  d88.  .d88b.  d8888b.
    88  `8D .8P  Y8. 88'          88'YbdP`88   `88'   888o  88 88'YbdP`88 .8P  Y8. 88  `8D
    88oobY' 88    88 88ooooo      88  88  88    88    88V8o 88 88  88  88 88    88 88   88
    88`8b   88    88 88~~~~~      88  88  88    88    88 V8o88 88  88  88 88    88 88   88
    88 `88. `8b  d8' 88.          88  88  88   .88.   88  V888 88  88  88 `8b  d8' 88  .8D
    88   YD  `Y88P'  Y88888P      YP  YP  YP Y888888P VP   V8P YP  YP  YP  `Y88P'  Y8888D'
  */

  // --x-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, XDIR, ROE>{
  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      IDIR;

      FLUXLIM(c,cm,cp,cmm,cpp, ROEMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, ROEMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, ROEMACRO);

    }
  };

  // --y-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, YDIR, ROE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      JDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,ROEMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, ROEMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, ROEMACRO);

    }
  };

  // --z-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, ZDIR, ROE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      KDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,ROEMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, ROEMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, ROEMACRO);

    }
  };

  /*
           db    db d8888b. db   d8b   db d888888b d8b   db d8888b.
           88    88 88  `8D 88   I8I   88   `88'   888o  88 88  `8D
           88    88 88oodD' 88   I8I   88    88    88V8o 88 88   88
           88    88 88~~~   Y8   I8I   88    88    88 V8o88 88   88
           88b  d88 88      `8b d8'8b d8'   .88.   88  V888 88  .8D
           ~Y8888P' 88       `8b8' `8d8'  Y888888P VP   V8P Y8888D'
  */

  // --x-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, XDIR, UPWIND>{
  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      IDIR;

      FLUXLIM(c,cm,cp,cmm,cpp, UPWINDMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, UPWINDMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, UPWINDMACRO);

    }
  };

  // --y-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, YDIR, UPWIND>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      JDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,UPWINDMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, UPWINDMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, UPWINDMACRO);

    }
  };

  // --z-dir--
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, ZDIR, UPWIND>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      KDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,UPWINDMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp,UPWINDMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp,UPWINDMACRO);

    }
  };

  /*
              .o88b. d88888b d8b   db d888888b d8888b.  .d8b.  db
             d8P  Y8 88'     888o  88 `~~88~~' 88  `8D d8' `8b 88
             8P      88ooooo 88V8o 88    88    88oobY' 88ooo88 88
             8b      88~~~~~ 88 V8o88    88    88`8b   88~~~88 88
             Y8b  d8 88.     88  V888    88    88 `88. 88   88 88booo.
              `Y88P' Y88888P VP   V8P    YP    88   YD YP   YP Y88888P
  */

  // x-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, XDIR, CENTRAL>{
  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      IDIR;

      FLUXLIM(c,cm,cp,cmm,cpp, CENTRALMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, CENTRALMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      IDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, CENTRALMACRO);

    }
  };

  // y-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, YDIR, CENTRAL>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      JDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,CENTRALMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, CENTRALMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      JDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp, CENTRALMACRO);

    }
  };

  // z-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, ZDIR, CENTRAL>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    UT& af;
    double A;
    const double huge;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      KDIR;

      FLUXLIM(c,cm,cp,cmm,cpp,CENTRALMACRO);

    }

    void
    operator()(TagPlusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp,CENTRALMACRO);

    }

    void
    operator()(TagMinusBC obj, int i, int j, int k ) const {

      KDIR;

      FLUXLIM_PLUSBC(c,cm,cp,cmm,cpp,CENTRALMACRO);

    }
  };

  //FACE CENTERED VARIABLES -------------------------------------------------------------

  // SFCXVariable<double>
  //
  // x-dir
  template<typename UT>
  struct ComputeConvection<SFCXVariable<double>, UT, XDIR, CENTRAL>{

    typedef typename VariableHelper<SFCXVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCXVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      XIDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // y-dir
  template<typename UT>
  struct ComputeConvection<SFCXVariable<double>, UT, YDIR, CENTRAL>{

    typedef typename VariableHelper<SFCXVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCXVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YIDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // z-dir
  template<typename UT>
  struct ComputeConvection<SFCXVariable<double>, UT, ZDIR, CENTRAL>{

    typedef typename VariableHelper<SFCXVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCXVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZIDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // SFCYVariable<double>
  //
  // x-dir
  template<typename UT>
  struct ComputeConvection<SFCYVariable<double>, UT, XDIR, CENTRAL>{

    typedef typename VariableHelper<SFCYVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCYVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YIDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // y-dir
  template<typename UT>
  struct ComputeConvection<SFCYVariable<double>, UT, YDIR, CENTRAL>{

    typedef typename VariableHelper<SFCYVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCYVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YJDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // z-dir
  template<typename UT>
  struct ComputeConvection<SFCYVariable<double>, UT, ZDIR, CENTRAL>{

    typedef typename VariableHelper<SFCYVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCYVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YKDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // SFCZVariable<double>
  //
  // x-dir
  template<typename UT>
  struct ComputeConvection<SFCZVariable<double>, UT, XDIR, CENTRAL>{

    typedef typename VariableHelper<SFCZVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCZVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZIDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // y-dir
  template<typename UT>
  struct ComputeConvection<SFCZVariable<double>, UT, YDIR, CENTRAL>{

    typedef typename VariableHelper<SFCZVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCZVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZJDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };

  // z-dir
  template<typename UT>
  struct ComputeConvection<SFCZVariable<double>, UT, ZDIR, CENTRAL>{

    typedef typename VariableHelper<SFCZVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCZVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    UT& af;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, UT& af, double A)
      : phi(phi), rhs(rhs), u(u), af(af), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZKDIR;

      CENTRALFACE(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);

    }
  };
}
#endif
