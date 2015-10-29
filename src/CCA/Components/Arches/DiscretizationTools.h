#ifndef Uintah_Component_Arches_DISCRETIZATIONTOOLS_h
#define Uintah_Component_Arches_DISCRETIZATIONTOOLS_h

namespace Uintah{

  //Helpers for variable types:
  template <typename T>
  struct VariableHelper{
  };

  //Helper specialization:
  template <>
  struct VariableHelper<CCVariable<double> >{
    typedef constCCVariable<double> ConstType;
    typedef CCVariable<double> Type;
  };

  template <>
  struct VariableHelper<SFCXVariable<double> >{
    typedef constSFCXVariable<double> ConstType;
    typedef SFCXVariable<double> Type;
  };

  template <>
  struct VariableHelper<SFCYVariable<double> >{
    typedef constSFCYVariable<double> ConstType;
    typedef SFCYVariable<double> Type;
  };

  template <>
  struct VariableHelper<SFCZVariable<double> >{
    typedef constSFCZVariable<double> ConstType;
    typedef SFCZVariable<double> Type;
  };

  //------------------------------------------------------------------------------------------------
  //intialization
  template <typename T>
  struct VariableInitializeFunctor{

    T& var;
    constCCVariable<double>& gridX;
    double value;

    VariableInitializeFunctor( T& var, constCCVariable<double>& gridX, double value ) 
      : var(var), gridX(gridX), value(value){}

    void operator()(int i, int j, int k) const{

      const IntVector c(i,j,k);
      double start = 0.5; 

      double value_assign = (gridX[c] > start) ? 0.0 : value; 

      var[c] = value_assign;

    }
  };

  //------------------------------------------------------------------------------------------------
  //convection
  template<typename PT, typename UT>
  struct ComputeConvection{
  };

  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    constCCVariable<double>& rho;
    UT& u;
    IntVector dir;
    double A;
    const double huge; 

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      IntVector c(i,j,k);
      IntVector cm(i-dir[0],j-dir[1],k-dir[2]);
      IntVector cp(i+dir[0],j+dir[1],k+dir[2]);
      IntVector cmm(i-2*dir[0],j-2*dir[1],k-2*dir[2]);
      IntVector cpp(i+2*dir[0],j+2*dir[1],k+2*dir[2]);

      double Sup_up = phi[cm];
      double Sdn_up = phi[c];
      double r = (phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] );

      double psi_up = ( r < huge ) ? std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) ) : 2.0;
      psi_up = std::max( 0.0, psi_up );

      double Sup_dn = phi[c];
      double Sdn_dn = phi[cm];
      double r2 = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] );

      double psi_dn = ( r2 < huge ) ? std::max( std::min( 2.*r2, 1.0), std::min(r2, 2.0 ) ) : 2.0;
      psi_dn = std::max( 0.0, psi_dn );

      double face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up );
      double face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn );

      double face_value_m = ( u[c] > 0.0 ) ? face_up : face_dn;

      Sup_up = phi[c];
      Sdn_up = phi[cp];
      double r3 = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] );

      psi_up = ( r3 < huge ) ? std::max( std::min( 2.*r3, 1.0), std::min(r3, 2.0 ) ) : 2.0;
      psi_up = std::max( 0.0, psi_up );

      Sup_dn = phi[cp];
      Sdn_dn = phi[c];
      double r4 =  ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] );

      psi_dn = ( r4 < huge ) ? std::max( std::min( 2.*r4, 1.0), std::min(r4, 2.0 ) ) : 2.0;
      psi_dn = std::max( 0.0, psi_dn );

      face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up );
      face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn );

      double face_value_p = ( u[cp] > 0.0 ) ? face_up : face_dn;

      //Done with interpolation, now compute conv and add to RHS:
      rhs[c] += -A * ( face_value_p * u[cp] - face_value_m * u[c] );

    }
  };

  template<typename UT>
  struct ComputeConvection<SFCXVariable<double>, UT>{

    typedef typename VariableHelper<SFCXVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCXVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    constCCVariable<double>& rho;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ) const{

      IntVector c(i,j,k);
      IntVector cm(i-dir[0],j-dir[1],k-dir[2]);
      IntVector cp(i+dir[0],j+dir[1],k+dir[2]);

      IntVector cu_w(i,j,k);
      IntVector cu_w2(i-1,j,k);

      IntVector cu_e(i,j+dir[1],k+dir[2]);
      IntVector cu_e2(i-1+dir[0]*2,j+dir[1],k+dir[2]);


      double phi_e = ( phi[c] + phi[cp] )/ 2.;
      double phi_w = ( phi[c] + phi[cm] )/ 2.;

      double u_e = ( u[cu_e] + u[cu_e2] ) / 2.;
      double u_w = ( u[cu_w] + u[cu_w2] ) / 2.;

      rhs[c] += -A * ( phi_e * u_e - phi_w * u_w );

    }
  };

  template<typename UT>
  struct ComputeConvection<SFCYVariable<double>, UT>{

    typedef typename VariableHelper<SFCYVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCYVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    constCCVariable<double>& rho;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ) const{

      IntVector c(i,j,k);
      IntVector cm(i-dir[0],j-dir[1],k-dir[2]);
      IntVector cp(i+dir[0],j+dir[1],k+dir[2]);

      IntVector cu_w(i,j,k);
      IntVector cu_w2(i,j-1,k);

      IntVector cu_e(i+dir[0],j,k+dir[2]);
      IntVector cu_e2(i+dir[0],j-1+dir[1]*2,k+dir[2]);


      double phi_e = ( phi[c] + phi[cp] )/ 2.;
      double phi_w = ( phi[c] + phi[cm] )/ 2.;

      double u_e = ( u[cu_e] + u[cu_e2] ) / 2.;
      double u_w = ( u[cu_w] + u[cu_w2] ) / 2.;

      rhs[c] += -A * ( phi_e * u_e - phi_w * u_w );

    }
  };

  template<typename UT>
  struct ComputeConvection<SFCZVariable<double>, UT>{

    typedef typename VariableHelper<SFCZVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCZVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    constCCVariable<double>& rho;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( ConstPT& phi, PT& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ) const{

      IntVector c(i,j,k);
      IntVector cm(i-dir[0],j-dir[1],k-dir[2]);
      IntVector cp(i+dir[0],j+dir[1],k+dir[2]);

      IntVector cu_w(i,j,k);
      IntVector cu_w2(i,j,k-1);

      IntVector cu_e(i+dir[0],j+dir[1],k);
      IntVector cu_e2(i+dir[0],j+dir[1],k-1+dir[2]*2);


      double phi_e = ( phi[c] + phi[cp] )/ 2.;
      double phi_w = ( phi[c] + phi[cm] )/ 2.;

      double u_e = ( u[cu_e] + u[cu_e2] ) / 2.;
      double u_w = ( u[cu_w] + u[cu_w2] ) / 2.;

      rhs[c] += -A * ( phi_e * u_e - phi_w * u_w );

    }
  };

  //------------------------------------------------------------------------------------------------
  //diffusion
  template <typename T>
  struct ComputeDiffusion{

    //for now assuming that this will only be used for CC variables:
    typedef typename VariableHelper<T>::ConstType ConstPT;
    typedef typename VariableHelper<T>::Type PT;

    ConstPT& phi;
    constCCVariable<double>& gamma; //!!!!!!!!
    PT& rhs;
    IntVector dir;
    double A;
    double dx;

    ComputeDiffusion( ConstPT& phi, constCCVariable<double>& gamma, PT& rhs, IntVector dir, double A, double dx ) :
      phi(phi), gamma(gamma), rhs(rhs), dir(dir), A(A), dx(dx) { }

    void operator()(int i, int j, int k) const {

      IntVector c(i,j,k);
      IntVector cm(i-dir[0],j-dir[1],k-dir[2]);
      IntVector cp(i+dir[0],j+dir[1],k+dir[2]);

      double face_gamma_e = (gamma[c] + gamma[cp])/ 2.0;
      double face_gamma_w = (gamma[c] + gamma[cm])/ 2.0;

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
