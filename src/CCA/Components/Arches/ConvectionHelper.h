#ifndef Uintah_Component_Arches_CONVECTIONHELPER_h
#define Uintah_Component_Arches_CONVECTIONHELPER_h

#include <CCA/Components/Arches/GridTools.h>
#include <cmath>
#include <sci_defs/kokkos_defs.h>

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
   a) Define the appropriate macros for the operator (e.g., SUPERBEEMACRO, VANLEERMACRO, ...)
   b) specialize ComputeConvection for the supported grid-variabe types. Note that
   the default template will be called for non-supported types and throw and error
   in the actual operator() if this is ever hit during run-time.
   c) potentially define a convenience macro (e.g., UPWIND_CONVECTION) to run through
   the three different directions is applicable.
*/

namespace Uintah {

  enum LIMITER { NOCONV, CENTRAL, UPWIND, SUPERBEE, ROE, VANLEER, FOURTH };

#define SUPERBEEMACRO(r) \
  my_psi = ( r < huge ) ? max( min( 2.*r, 1.), min(r, 2. ) ) : 2.; \
  my_psi = max( 0., my_psi );

#define ROEMACRO(r) \
  my_psi = ( r < huge ) ? min(1., r) : 1.; \
  my_psi = max(0., my_psi);

#define VANLEERMACRO(r) \
  my_psi = ( r < huge ) ? ( r + fabs(r) ) / ( 1. + fabs(r) ) : 2.; \
  my_psi = ( r >= 0. ) ? my_psi : 0.;

  /**
      @struct IntegrateFlux
      @brief  Given a flux variable, integrate to get the total contribution to the RHS.
  **/
  template <typename T>
  struct IntegrateFlux {

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<CT>::XFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<CT>::YFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<CT>::ZFaceType CFZT;

//    typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType CFXT;
//    typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType CFYT;
//    typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType CFZT;

    IntegrateFlux(       T      & rhs
                 ,       CFXT   & flux_x
                 ,       CFYT   & flux_y
                 ,       CFZT   & flux_z
                 , const Vector & Dx
                 )
      :
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
        rhs( rhs.getKokkosView() )
      , flux_x( flux_x.getKokkosView() )
      , flux_y( flux_y.getKokkosView() )
      , flux_z( flux_z.getKokkosView() )
#else
        rhs( rhs )
      , flux_x( flux_x )
      , flux_y( flux_y )
      , flux_z( flux_z )
#endif
      , Dx( Dx )
    {}

    void operator()(int i, int j, int k) const
    {
      double ax = Dx.y() * Dx.z();
      double ay = Dx.z() * Dx.x();
      double az = Dx.x() * Dx.y();

      rhs(i,j,k) = -1. * ( ax * ( flux_x(i+1,j,k) - flux_x(i,j,k) ) +
                           ay * ( flux_y(i,j+1,k) - flux_y(i,j,k) ) +
                           az * ( flux_z(i,j,k+1) - flux_z(i,j,k) ) );
    }

  private:

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
    KokkosView3<      double, Kokkos::HostSpace> rhs;
    KokkosView3<const double, Kokkos::HostSpace> flux_x;
    KokkosView3<const double, Kokkos::HostSpace> flux_y;
    KokkosView3<const double, Kokkos::HostSpace> flux_z;
#else
    T    & rhs;
    CFXT & flux_x;
    CFYT & flux_y;
    CFZT & flux_z;
#endif
    const Vector & Dx;

  }; // struct IntegrateFlux

  /** @struct ComputeConvectiveFluxHelper **/
  enum convType { FourthConvection
                , UpwindConvection
                , CentralConvection
                , VanLeerConvection
                , RoeConvection
                , SuperBeeConvection
                };

  template <typename PSIX_T, typename PSIY_T, typename PSIZ_T, typename grid_T>
  struct ComputeConvectiveFlux4 {

    ComputeConvectiveFlux4( const grid_T & i_phi
                          , const grid_T & i_u
                          , const grid_T & i_v
                          , const grid_T & i_w
                          ,       PSIX_T & i_psi_x
                          ,       PSIY_T & i_psi_y
                          ,       PSIZ_T & i_psi_z
                          ,       grid_T & i_flux_x
                          ,       grid_T & i_flux_y
                          ,       grid_T & i_flux_z
                          , const grid_T & i_eps
                          )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , psi_x( i_psi_x )
      , psi_y( i_psi_y )
      , psi_z( i_psi_z )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    void operator()(int i, int j, int k ) const
    {
      double c1 = 7./12.;
      double c2 = -1./12.;

      //std::cout<<"fourth convection"<< std::endl;

      //X-dir
      {
        STENCIL5_1D(0);

        const double afc  =  eps(IJK_) * eps(IJK_M_) ;

        flux_x(IJK_) = afc * u(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
      }

      //Y-dir
      {
        STENCIL5_1D(1);

        const double afc  =  eps(IJK_) * eps(IJK_M_) ;

        flux_y(IJK_) = afc * v(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
      }

      //Z-dir
      {
        STENCIL5_1D(2);

        const double afc  =  eps(IJK_) * eps(IJK_M_) ;

        flux_z(IJK_) = afc * w(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
      }
    }

  private:

    const grid_T & phi;
    const grid_T & u;
    const grid_T & v;
    const grid_T & w;
          PSIX_T & psi_x;
          PSIY_T & psi_y;
          PSIZ_T & psi_z;
          grid_T & flux_x;
          grid_T & flux_y;
          grid_T & flux_z;
    const grid_T & eps;

  }; // struct ComputeConvectiveFlux4

  /**
      @struct ComputeConvectiveFlux
      @brief Compute a convective flux given psi (flux limiter) with this functor.
             The template arguments arrise from the potential use of temporary (non-const)
             variables. Typically, one would use T=grid_T and CT=const grid_T
             when not using temporary variables. However, since temporary variables can ONLY
             be non-const, one should use CT=grid_T in that case.
  **/
  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT, unsigned int Cscheme>
  struct ComputeConvectiveFlux1D {

    void get_flux( ExecutionObject<ExecSpace, MemSpace>   execObj
                 , BlockRange                           & range
                 , grid_CT                              & phi
                 , grid_CT                              & u
                 , grid_T                               & flux
                 , grid_CT                              & eps
                 , int                                    dir
                 )
    {
      throw InvalidValue("Error: Convection scheme not valid.",__FILE__, __LINE__);
    }
  }; // struct ComputeConvectiveFlux1D

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, UpwindConvection> {

    void get_flux( ExecutionObject<ExecSpace, MemSpace>   execObj
                 , BlockRange                           & range
                 , grid_CT                              & phi
                 , grid_CT                              & u
                 , grid_T                               & flux
                 , grid_CT                              & eps
                 , int                                    dir
                 )
    {
      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        STENCIL3_1D(dir);

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double afc = eps(IJK_)*eps(IJK_M_);

        flux(IJK_) = afc * u(IJK_) * Sup;
      });
    }
  }; // struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, UpwindConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, CentralConvection> {

    void get_flux( ExecutionObject<ExecSpace, MemSpace>   execObj
                 , BlockRange                           & range
                 , grid_CT                              & phi
                 , grid_CT                              & u
                 , grid_T                               & flux
                 , grid_CT                              & eps
                 , int                                    dir
                 )
    {
      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        STENCIL3_1D(dir);

        const double afc = eps(IJK_)*eps(IJK_M_);

        flux(IJK_) = afc * u(IJK_) * 0.5 * ( phi(IJK_) + phi(IJK_M_));
      });
    }
  }; // struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, CentralConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, SuperBeeConvection> {

    void get_flux( ExecutionObject<ExecSpace, MemSpace>   execObj
                 , BlockRange                           & range
                 , grid_CT                              & phi
                 , grid_CT                              & u
                 , grid_T                               & flux
                 , grid_CT                              & eps
                 , int                                    dir
                 )
    {
      const double tiny = 1.0e-16;
      const double huge = 1.0e10;

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        double my_psi;

        STENCIL5_1D(dir);

        double r = u(IJK_) > 0 ? fabs(( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny )) :
                                 fabs(( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ));

        SUPERBEEMACRO(r);

        const double afc  = eps(IJK_)* eps(IJK_M_);
        const double afcm = eps(IJK_M_)* eps(IJK_MM_);

        my_psi *= afc * afcm;

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        my_psi *= ( Sdn - Sup ); //split flux calc in two lines to avoid floating point difference in CPU and GPU

        flux(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi );
      });
    }
  }; // struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, SuperBeeConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, VanLeerConvection> {

    void get_flux( ExecutionObject<ExecSpace, MemSpace>   execObj
                 , BlockRange                           & range
                 , grid_CT                              & phi
                 , grid_CT                              & u
                 , grid_T                               & flux
                 , grid_CT                              & eps
                 , int                                    dir
                 )
    {
      const double tiny = 1.0e-16;
      const double huge = 1.0e10;

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        double my_psi;

        STENCIL5_1D(dir);

        double r = u(IJK_) > 0 ? fabs(( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny )) :
                                 fabs(( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ));

        VANLEERMACRO(r);

        const double afc  = eps(IJK_)* eps(IJK_M_);
        const double afcm = eps(IJK_M_)* eps(IJK_MM_);

        my_psi *= afc * afcm;

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        my_psi *= ( Sdn - Sup ); //split flux calc in two lines to avoid floating point difference in CPU and GPU

        flux(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi);
      });
    }
  }; // struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, VanLeerConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, RoeConvection> {

    void get_flux( ExecutionObject<ExecSpace, MemSpace>   execObj
                 , BlockRange                           & range
                 , grid_CT                              & phi
                 , grid_CT                              & u
                 , grid_T                               & flux
                 , grid_CT                              & eps
                 , int                                    dir
                 )
    {
      const double tiny = 1.0e-16;
      const double huge = 1.0e10;

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        double my_psi;

        STENCIL5_1D(dir);

        double r = u(IJK_) > 0 ? fabs(( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny )) :
                                 fabs(( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ));

        ROEMACRO(r);

        const double afc  = eps(IJK_)* eps(IJK_M_);
        const double afcm = eps(IJK_M_)* eps(IJK_MM_);

        my_psi *= afc * afcm;

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      });
    }
  }; // struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, RoeConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, FourthConvection> {

    void get_flux( ExecutionObject<ExecSpace, MemSpace>   execObj
                 , BlockRange                           & range
                 , grid_CT                              & phi
                 , grid_CT                              & u
                 , grid_T                               & flux
                 , grid_CT                              & eps
                 , int                                    dir
                 )
    {
      const double c1{7./12.};
      const double c2{-1./12.};

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        STENCIL5_1D(dir);

        const double afc  = eps(IJK_)* eps(IJK_M_) ;

        flux(IJK_) = afc * u(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
      });
    }
  }; // struct ComputeConvectiveFlux1D<ExecSpace, MemSpace, grid_T, grid_CT, FourthConvection>

  template <typename grid_T, typename grid_CT, unsigned int Cscheme>
  struct ComputeConvectiveFlux {

    ComputeConvectiveFlux( const grid_CT & i_phi
                         , const grid_CT & i_u
                         , const grid_CT & i_v
                         , const grid_CT & i_w
                         ,       grid_T  & i_flux_x
                         ,       grid_T  & i_flux_y
                         ,       grid_T  & i_flux_z
                         , const grid_CT & i_eps
                         )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    // Default operator - throw an error
    void operator()( int i, int j, int k ) const
    {
      throw InvalidValue("Error: Convection scheme not valid.",__FILE__, __LINE__);
    }
 
  private:

    const grid_CT & phi;
    const grid_CT & u;
    const grid_CT & v;
    const grid_CT & w;
          grid_T  & flux_x;
          grid_T  & flux_y;
          grid_T  & flux_z;
    const grid_CT & eps;

  }; // struct ComputeConvectiveFlux

  template <typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux<grid_T, grid_CT, UpwindConvection> {

    ComputeConvectiveFlux( const grid_CT & i_phi
                         , const grid_CT & i_u
                         , const grid_CT & i_v
                         , const grid_CT & i_w
                         ,       grid_T  & i_flux_x
                         ,       grid_T  & i_flux_y
                         ,       grid_T  & i_flux_z
                         , const grid_CT & i_eps
                         )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    void operator()(  int i, int j, int k ) const
    {
      //X-dir
      {
        STENCIL3_1D(0);

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double afc = eps(IJK_)*eps(IJK_M_);

        flux_x(IJK_) = afc * u(IJK_) * Sup;
      }

      //Y-dir
      {
        STENCIL3_1D(1);

        const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double afc = eps(IJK_)*eps(IJK_M_);

        flux_y(IJK_) = afc * v(IJK_) * Sup;
      }

      //Z-dir
      {
        STENCIL3_1D(2);

        const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double afc = eps(IJK_)*eps(IJK_M_);

        flux_z(IJK_) = afc * w(IJK_) * Sup;
      }
    }

  private:

    const grid_CT & phi;
    const grid_CT & u;
    const grid_CT & v;
    const grid_CT & w;
          grid_T  & flux_x;
          grid_T  & flux_y;
          grid_T  & flux_z;
    const grid_CT & eps;

  }; // struct ComputeConvectiveFlux<grid_T, grid_CT, UpwindConvection>

  template <typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux<grid_T, grid_CT, CentralConvection> {

    ComputeConvectiveFlux( const grid_CT & i_phi
                         , const grid_CT & i_u
                         , const grid_CT & i_v
                         , const grid_CT & i_w
                         ,       grid_T  & i_flux_x
                         ,       grid_T  & i_flux_y
                         ,       grid_T  & i_flux_z
                         , const grid_CT & i_eps
                         )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    void operator()( int i, int j, int k ) const
    {

      //X-dir
      {
        STENCIL3_1D(0);

        const double afc = eps(IJK_)*eps(IJK_M_);

        flux_x(IJK_) = afc * u(IJK_) * 0.5 * ( phi(IJK_) + phi(IJK_M_));
      }

      //Y-dir
      {
        STENCIL3_1D(1);

        const double afc = eps(IJK_)*eps(IJK_M_);

        flux_y(IJK_) = afc * v(IJK_) * 0.5 * ( phi(IJK_) + phi(IJK_M_));
      }

      //Z-dir
      {
        STENCIL3_1D(2);

        const double afc = eps(IJK_)*eps(IJK_M_);

        flux_z(IJK_) = afc * w(IJK_) * 0.5 * ( phi(IJK_) + phi(IJK_M_));
      }
    }

  private:

    const grid_CT & phi;
    const grid_CT & u;
    const grid_CT & v;
    const grid_CT & w;
          grid_T  & flux_x;
          grid_T  & flux_y;
          grid_T  & flux_z;
    const grid_CT & eps;

  }; // struct ComputeConvectiveFlux<grid_T, grid_CT, CentralConvection>

  template <typename grid_T , typename grid_CT>
  struct ComputeConvectiveFlux<grid_T, grid_CT, SuperBeeConvection> {

    ComputeConvectiveFlux( const grid_CT & i_phi
                         , const grid_CT & i_u
                         , const grid_CT & i_v
                         , const grid_CT & i_w
                         ,       grid_T  & i_flux_x
                         ,       grid_T  & i_flux_y
                         ,       grid_T  & i_flux_z
                         , const grid_CT & i_eps
                         )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    void operator()( int i, int j, int k ) const
    {
      const double tiny = 1.0e-16;
      const double huge = 1.0e10;

      //X-dir
      {
        double my_psi;

        STENCIL5_1D(0);

        const double r = u(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        SUPERBEEMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_x(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }

      //Y-dir
      {
        double my_psi;

        STENCIL5_1D(1);

        const double r = v(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        SUPERBEEMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = v(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_y(IJK_) = afc * v(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }

      //Z-dir
      {
        double my_psi;

        STENCIL5_1D(2);

        const double r = w(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        SUPERBEEMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = w(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_z(IJK_) = afc * w(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }
    }

  private:

    const grid_CT & phi;
    const grid_CT & u;
    const grid_CT & v;
    const grid_CT & w;
          grid_T  & flux_x;
          grid_T  & flux_y;
          grid_T  & flux_z;
    const grid_CT & eps;

  }; // struct ComputeConvectiveFlux<grid_T, grid_CT, SuperBeeConvection>

  template <typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux<grid_T, grid_CT, VanLeerConvection> {

    ComputeConvectiveFlux( const grid_CT & i_phi
                         , const grid_CT & i_u
                         , const grid_CT & i_v
                         , const grid_CT & i_w
                         ,       grid_T  & i_flux_x
                         ,       grid_T  & i_flux_y
                         ,       grid_T  & i_flux_z
                         , const grid_CT & i_eps
                         )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    void operator()(  int i, int j, int k ) const
    {
      const double tiny = 1.0e-16;
      const double huge = 1.0e10;

      //X-dir
      {
        double my_psi;

        STENCIL5_1D(0);

        const double r = u(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        VANLEERMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_x(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }

      //Y-dir
      {
        double my_psi;

        STENCIL5_1D(1);

        const double r = v(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        VANLEERMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = v(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_y(IJK_) = afc * v(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }

      //Z-dir
      {
        double my_psi;

        STENCIL5_1D(2);

        const double r = w(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        VANLEERMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = w(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_z(IJK_) = afc * w(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }
    }

  private:

    const grid_CT & phi;
    const grid_CT & u;
    const grid_CT & v;
    const grid_CT & w;
          grid_T  & flux_x;
          grid_T  & flux_y;
          grid_T  & flux_z;
    const grid_CT & eps;

  }; // struct ComputeConvectiveFlux<grid_T, grid_CT, VanLeerConvection>

  template <typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux<grid_T, grid_CT, RoeConvection> {

    ComputeConvectiveFlux( const grid_CT & i_phi
                         , const grid_CT & i_u
                         , const grid_CT & i_v
                         , const grid_CT & i_w
                         ,       grid_T  & i_flux_x
                         ,       grid_T  & i_flux_y
                         ,       grid_T  & i_flux_z
                         , const grid_CT & i_eps
                         )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    void operator()( int i, int j, int k ) const
    {
      //X-dir
      {
        double my_psi;

        STENCIL5_1D(0);

        const double r = u(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        ROEMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_x(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }

      //Y-dir
      {
        double my_psi;

        STENCIL5_1D(1);

        const double r = v(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        ROEMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = v(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_y(IJK_) = afc * v(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }

      //Z-dir
      {
        double my_psi;

        STENCIL5_1D(2);

        const double r = w(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                       fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

        ROEMACRO(r);

        const double afc = eps(IJK_)*eps(IJK_M_);
        const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

        my_psi *= afc * afcm;

        const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
        const double Sdn = w(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

        flux_z(IJK_) = afc * w(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
      }
    }

  private:

    const grid_CT & phi;
    const grid_CT & u;
    const grid_CT & v;
    const grid_CT & w;
          grid_T  & flux_x;
          grid_T  & flux_y;
          grid_T  & flux_z;
    const grid_CT & eps;
    const double    tiny{1.0e-16};
    const double    huge{1.0e10};

  }; // struct ComputeConvectiveFlux<grid_T, grid_CT, RoeConvection>

  template <typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux<grid_T, grid_CT, FourthConvection> {

    ComputeConvectiveFlux( const grid_CT & i_phi
                         , const grid_CT & i_u
                         , const grid_CT & i_v
                         , const grid_CT & i_w
                         ,       grid_T  & i_flux_x
                         ,       grid_T  & i_flux_y
                         ,       grid_T  & i_flux_z
                         , const grid_CT & i_eps
                         )
      : phi( i_phi )
      , u( i_u )
      , v( i_v )
      , w( i_w )
      , flux_x( i_flux_x )
      , flux_y( i_flux_y )
      , flux_z( i_flux_z )
      , eps( i_eps )
    {}

    // Default operator - throw an error
    void operator()( int i, int j, int k ) const
    {
      //X-dir
      {
        STENCIL5_1D(0);

        const double afc  =  eps(IJK_) * eps(IJK_M_) ;

        flux_x(IJK_) = afc * u(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) ) ;
      }

      //Y-dir
      {
        STENCIL5_1D(1);

        const double afc  =  eps(IJK_) * eps(IJK_M_) ;

        flux_y(IJK_) = afc * v(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
      }

      //Z-dir
      {
        STENCIL5_1D(2);

        const double afc  =  eps(IJK_) * eps(IJK_M_) ;

        flux_z(IJK_) = afc * w(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
      }
    }

  private:

    const grid_CT & phi;
    const grid_CT & u;
    const grid_CT & v;
    const grid_CT & w;
          grid_T  & flux_x;
          grid_T  & flux_y;
          grid_T  & flux_z;
    const grid_CT & eps;
    const double    c1 {7./12.};
    const double    c2 {-1./12.};

  }; // struct ComputeConvectiveFlux<grid_T, grid_CT, FourthConvection>

  //====================================================================================
  // WARNING: CODE DUPLICATION
  //====================================================================================

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT, unsigned int Cscheme>
  struct ComputeConvectiveFlux3D {

    void get_flux(       ExecutionObject<ExecSpace, MemSpace> & execObj
                 ,       BlockRange                           & range
                 , const grid_CT                              & phi
                 , const grid_CT                              & u
                 , const grid_CT                              & v
                 , const grid_CT                              & w
                 ,       grid_T                               & flux_x
                 ,       grid_T                               & flux_y
                 ,       grid_T                               & flux_z
                 , const grid_CT                              & eps
                 )
      {
      throw InvalidValue(
        "Error: Convection scheme not valid.",__FILE__, __LINE__);
      }
  }; // struct ComputeConvectiveFlux3D

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, UpwindConvection> {

    void get_flux(       ExecutionObject<ExecSpace, MemSpace> & execObj
                 ,       BlockRange                           & range
                 , const grid_CT                              & phi
                 , const grid_CT                              & u
                 , const grid_CT                              & v
                 , const grid_CT                              & w
                 ,       grid_T                               & flux_x
                 ,       grid_T                               & flux_y
                 ,       grid_T                               & flux_z
                 , const grid_CT                              & eps
                 )
    {
      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        //X-dir
        {
          STENCIL3_1D(0);

          const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double afc = eps(IJK_)*eps(IJK_M_);

          flux_x(IJK_) = afc * u(IJK_) * Sup;
        }

        //Y-dir
        {
          STENCIL3_1D(1);

          const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double afc = eps(IJK_)*eps(IJK_M_);

          flux_y(IJK_) = afc * v(IJK_) * Sup;
        }

        //Z-dir
        {
          STENCIL3_1D(2);

          const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double afc = eps(IJK_)*eps(IJK_M_);

          flux_z(IJK_) = afc * w(IJK_) * Sup;
        }
      });
    }
  }; // struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, UpwindConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, CentralConvection> {

    void get_flux(       ExecutionObject<ExecSpace, MemSpace> & execObj
                 ,       BlockRange                           & range
                 , const grid_CT                              & phi
                 , const grid_CT                              & u
                 , const grid_CT                              & v
                 , const grid_CT                              & w
                 ,       grid_T                               & flux_x
                 ,       grid_T                               & flux_y
                 ,       grid_T                               & flux_z
                 , const grid_CT                              & eps
                 )
    {
      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        //X-dir
        {
          STENCIL3_1D(0);

          const double afc = eps(IJK_)*eps(IJK_M_);

          flux_x(IJK_) = afc * u(IJK_) * 0.5 * ( phi(IJK_) + phi(IJK_M_));
        }

        //Y-dir
        {
          STENCIL3_1D(1);

          const double afc = eps(IJK_)*eps(IJK_M_);

          flux_y(IJK_) = afc * v(IJK_) * 0.5 * ( phi(IJK_) + phi(IJK_M_));
        }

        //Z-dir
        {
          STENCIL3_1D(2);

          const double afc = eps(IJK_)*eps(IJK_M_);

          flux_z(IJK_) = afc * w(IJK_) * 0.5 * ( phi(IJK_) + phi(IJK_M_));
        }
      });
    }
  }; // struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, CentralConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, SuperBeeConvection> {

    void get_flux(       ExecutionObject<ExecSpace, MemSpace> & execObj
                 ,       BlockRange                           & range
                 , const grid_CT                              & phi
                 , const grid_CT                              & u
                 , const grid_CT                              & v
                 , const grid_CT                              & w
                 ,       grid_T                               & flux_x
                 ,       grid_T                               & flux_y
                 ,       grid_T                               & flux_z
                 , const grid_CT                              & eps
                 )
    {
      const double tiny = 1.0e-16;
      const double huge = 1.0e10;

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        //X-dir
        {
          double my_psi;

          STENCIL5_1D(0);

          const double r = u(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          SUPERBEEMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_x(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
        }

        //Y-dir
        {
          double my_psi;

          STENCIL5_1D(1);

          const double r = v(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          SUPERBEEMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = v(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_y(IJK_) = afc * v(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
        }

        //Z-dir
        {
          double my_psi;

          STENCIL5_1D(2);
          const double r = w(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          SUPERBEEMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = w(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_z(IJK_) = afc * w(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
        }
      });
    }
  }; // struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, SuperBeeConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, VanLeerConvection> {

    void get_flux(       ExecutionObject<ExecSpace, MemSpace> & execObj
                 ,       BlockRange                           & range
                 , const grid_CT                              & phi
                 , const grid_CT                              & u
                 , const grid_CT                              & v
                 , const grid_CT                              & w
                 ,       grid_T                               & flux_x
                 ,       grid_T                               & flux_y
                 ,       grid_T                               & flux_z
                 , const grid_CT                              & eps
                 )
    {
      const double tiny = 1.0e-16;
      const double huge = 1.0e10;

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        //X-dir
        {
          double my_psi;

          STENCIL5_1D(0);

          const double r = u(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          VANLEERMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_x(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
        }

        //Y-dir
        {
          double my_psi;

          STENCIL5_1D(1);

          const double r = v(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          VANLEERMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = v(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_y(IJK_) = afc * v(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
        }

        //Z-dir
        {
          double my_psi;

          STENCIL5_1D(2);

          const double r = w(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          VANLEERMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = w(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_z(IJK_) = afc * w(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
        }
      });
    }
  }; // struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, VanLeerConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, RoeConvection> {

    void get_flux(       ExecutionObject<ExecSpace, MemSpace> & execObj
                 ,       BlockRange                           & range
                 , const grid_CT                              & phi
                 , const grid_CT                              & u
                 , const grid_CT                              & v
                 , const grid_CT                              & w
                 ,       grid_T                               & flux_x
                 ,       grid_T                               & flux_y
                 ,       grid_T                               & flux_z
                 , const grid_CT                              & eps
                 )
    {
      const double tiny{1.0e-16};
      const double huge{1.0e10};

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        //X-dir
        {
          double my_psi;

          STENCIL5_1D(0);

          const double r = u(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          ROEMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = u(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = u(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_x(IJK_) = afc * u(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup )) ;
        }

        //Y-dir
        {
          double my_psi;

          STENCIL5_1D(1);

          const double r = v(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          ROEMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = v(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = v(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_y(IJK_) = afc * v(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup )) ;
        }

        //Z-dir
        {
          double my_psi;

          STENCIL5_1D(2);

          const double r = w(IJK_) > 0 ? fabs( ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) ):
                                         fabs( ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny ) );

          ROEMACRO(r);

          const double afc = eps(IJK_)*eps(IJK_M_);
          const double afcm =  eps(IJK_M_) * eps(IJK_MM_) ;

          my_psi *= afc * afcm;

          const double Sup = w(IJK_) > 0 ? phi(IJK_M_) : phi(IJK_);
          const double Sdn = w(IJK_) > 0 ? phi(IJK_) : phi(IJK_M_);

          flux_z(IJK_) = afc * w(IJK_) * ( Sup + 0.5 * my_psi * ( Sdn - Sup ));
        }
      });
    }
  }; // struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, RoeConvection>

  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, FourthConvection> {

    void get_flux(       ExecutionObject<ExecSpace, MemSpace> & execObj
                 ,       BlockRange                           & range
                 , const grid_CT                              & phi
                 , const grid_CT                              & u
                 , const grid_CT                              & v
                 , const grid_CT                              & w
                 ,       grid_T                               & flux_x
                 ,       grid_T                               & flux_y
                 ,       grid_T                               & flux_z
                 , const grid_CT                              & eps
                 )
    {
      const double c1 {7./12.};
      const double c2 {-1./12.};

      parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){

        //X-dir
        {
          STENCIL5_1D(0);

          const double afc  =  eps(IJK_) * eps(IJK_M_) ;

          flux_x(IJK_) = afc * u(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) ) ;
        }

        //Y-dir
        {
          STENCIL5_1D(1);

          const double afc  =  eps(IJK_) * eps(IJK_M_) ;

          flux_y(IJK_) = afc * v(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
        }

        //Z-dir
        {
          STENCIL5_1D(2);

          const double afc  =  eps(IJK_) * eps(IJK_M_) ;

          flux_z(IJK_) = afc * w(IJK_) * ( c1*(phi(IJK_) + phi(IJK_M_)) + c2*(phi(IJK_MM_) + phi(IJK_P_)) );
        }
      });
    }
  }; // struct ComputeConvectiveFlux3D<ExecSpace, MemSpace, grid_T, grid_CT, FourthConvection>

  /**
      @struct GetPsi
      @brief Interface for computing psi. If
      actually used, this function should throw an error since the specialized versions
      are the functors actually doing the work.
  **/
  template <typename grid_T, unsigned int Cscheme>
  struct GetPsi {

    GetPsi( const grid_T & i_phi
          ,       grid_T & i_psi
          , const grid_T & i_u
          , const grid_T & i_eps
          , const int      i_dir
          )
      : phi( i_phi )
      , u( i_u )
      , eps( i_eps )
      , psi( i_psi )
      , dir( i_dir )
      , huge( 1.e10 )
      , tiny( 1.e-32 )
    {}

    void operator()(int i, int j, int k) const
    {
      throw InvalidValue("Error: No implementation of this limiter type or direction in Arches.h",__FILE__, __LINE__);
    }

  private:

    const grid_T & phi;
    const grid_T & u;
    const grid_T & eps;
          grid_T & psi;
    const int      dir;
    const double   huge;
    const double   tiny;

  }; // struct GetPsi

  template <typename grid_T>
  struct GetPsi<grid_T, SuperBeeConvection> {

    GetPsi( const grid_T & i_phi
          ,       grid_T & i_psi
          , const grid_T & i_u
          , const grid_T & i_eps
          , const int      i_dir
          )
      : phi( i_phi )
      , u( i_u )
      , eps( i_eps )
      , psi( i_psi )
      , dir( i_dir )
      , huge( 1.e10 )
      , tiny( 1.e-32 )
    {}

    void operator()( int i, int j, int k) const
    {
      double my_psi;
      double r;

      STENCIL5_1D(dir);

      r = u(IJK_) > 0 ? ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) :
                        ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny );
      r = fabs(r);

      SUPERBEEMACRO(r);

      const double afc  = (( eps(IJK_) + eps(IJK_M_) )/2.) < 0.51 ? 0. : 1.;
      const double afcm = (( eps(IJK_M_) + eps(IJK_MM_) )/2.) < 0.51 ? 0. : 1.;

      psi(IJK_) = my_psi * afc * afcm;
    }

  private:

    const grid_T & phi;
    const grid_T & u;
    const grid_T & eps;
          grid_T & psi;
    const int      dir;
    const double   huge;
    const double   tiny;

  }; // struct GetPsi<grid_T, SuperBeeConvection>

  template <typename grid_T>
  struct GetPsi<grid_T, RoeConvection> {

    GetPsi( const grid_T & i_phi
          ,       grid_T & i_psi
          , const grid_T & i_u
          , const grid_T & i_eps
          , const int      i_dir
          )
      : phi( i_phi )
      , u( i_u )
      , eps( i_eps )
      , psi( i_psi )
      , dir( i_dir )
      , huge( 1.e10 )
      , tiny( 1.e-32 )
    {}

    void operator()( int i, int j, int k) const
    {
      double my_psi;
      double r;

      STENCIL5_1D(dir);

      r = u(IJK_) > 0 ? ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) :
                        ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny );
      r = fabs(r);

      ROEMACRO(r);

      const double afc  = (( eps(IJK_) + eps(IJK_M_) )/2.) < 0.51 ? 0. : 1.;
      const double afcm = (( eps(IJK_M_) + eps(IJK_MM_) )/2.) < 0.51 ? 0. : 1.;

      psi(IJK_) = my_psi * afc * afcm;
    }

  private:

    const grid_T & phi;
    const grid_T & u;
    const grid_T & eps;
          grid_T & psi;
    const int      dir;
    const double   huge;
    const double   tiny;

  }; // struct GetPsi<grid_T, RoeConvection>

  template <typename grid_T>
  struct GetPsi<grid_T, VanLeerConvection> {

    GetPsi( const grid_T & i_phi
          ,       grid_T & i_psi
          , const grid_T & i_u
          , const grid_T & i_eps
          , const int      i_dir
          )
      : phi( i_phi )
      , u( i_u )
      , eps( i_eps )
      , psi( i_psi )
      , dir( i_dir )
      , huge( 1.e10 )
      , tiny( 1.e-32 )
    {}

    void operator()( int i, int j, int k) const
    {
      double my_psi;
      double r;

      STENCIL5_1D(dir);

      r = u(IJK_) > 0 ? ( phi(IJK_M_) - phi(IJK_MM_) ) / ( phi(IJK_) - phi(IJK_M_) + tiny ) :
                        ( phi(IJK_) - phi(IJK_P_) ) / ( phi(IJK_M_) - phi(IJK_) + tiny );
      r = fabs(r);

      VANLEERMACRO(r);

      const double afc  = (( eps(IJK_) + eps(IJK_M_) )/2.) < 0.51 ? 0. : 1.;
      const double afcm = (( eps(IJK_M_) + eps(IJK_MM_) )/2.) < 0.51 ? 0. : 1.;

      psi(IJK_) = my_psi * afc * afcm;
    }

  private:

    const grid_T & phi;
    const grid_T & u;
    const grid_T & eps;
          grid_T & psi;
    const int      dir;
    const double   huge;
    const double   tiny;

  }; // struct GetPsi<grid_T, VanLeerConvection>

  template <typename grid_T>
  struct GetPsi<grid_T, UpwindConvection> {

    GetPsi( const grid_T & i_phi
          ,       grid_T & i_psi
          , const grid_T & i_u
          , const grid_T & i_eps
          , const int      i_dir
          )
      : phi( i_phi )
      , u( i_u )
      , eps( i_eps )
      , psi( i_psi )
      , dir( i_dir )
      , huge( 1.e10 )
      , tiny( 1.e-32 )
    {}

    void operator()( int i, int j, int k) const
    {
      psi(IJK_) = 0.;
    }

  private:

    const grid_T & phi;
    const grid_T & u;
    const grid_T & eps;
          grid_T & psi;
    const int      dir;
    const double   huge;
    const double   tiny;

  }; // struct GetPsi<grid_T, UpwindConvection>

  template <typename grid_T>
  struct GetPsi<grid_T, CentralConvection> {

    GetPsi( const grid_T & i_phi
          ,       grid_T & i_psi
          , const grid_T & i_u
          , const grid_T & i_eps
          , const int      i_dir
          )
      : phi( i_phi )
      , u( i_u )
      , eps( i_eps )
      , psi( i_psi )
      , dir( i_dir )
      , huge( 1.e10 )
      , tiny( 1.e-32 )
    {}

    void operator()(int i, int j, int k) const
    {
      psi(IJK_) = 1.;
    }

  private:

    const grid_T & phi;
    const grid_T & u;
    const grid_T & eps;
          grid_T & psi;
    const int      dir;
    const double   huge;
    const double   tiny;

  }; // struct GetPsi<grid_T, CentralConvection>

  /**
    @class ConvectionHelper
    @brief A set of useful tools
  **/
  class ConvectionHelper {

  public:
    ConvectionHelper(){}
    ~ConvectionHelper(){}

    /**
      @brief Get the limiter enum from a string representation
    **/
    LIMITER get_limiter_from_string( const std::string value )
    {
      if ( value == "central" ){
        return CENTRAL;
      }
      else if ( value == "fourth" ){
        return FOURTH;
      }
      else if ( value == "upwind" ){
        return UPWIND;
      }
      else if ( value == "superbee" ){
        return SUPERBEE;
      }
      else if ( value == "roe" ){
        return ROE;
      }
      else if ( value == "vanleer" ){
        return VANLEER;
      }
      else {
        throw InvalidValue("Error: flux limiter type not recognized: "+value, __FILE__, __LINE__);
      }
    }

  }; // class ConvectionHelper

} // namespace Uintah
#endif
