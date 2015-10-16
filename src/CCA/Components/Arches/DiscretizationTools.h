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
  };

  template <>
  struct VariableHelper<SFCXVariable<double> >{
    typedef constSFCXVariable<double> ConstType;
  };

  template <>
  struct VariableHelper<SFCYVariable<double> >{
    typedef constSFCYVariable<double> ConstType;
  };

  template <>
  struct VariableHelper<SFCZVariable<double> >{
    typedef constSFCZVariable<double> ConstType;
  };

  //intialization
  template <typename T>
  struct VariableInitializeFunctor{

    T& var;
    double value;

    VariableInitializeFunctor( T& var, double value ) : var(var), value(value){}

    void operator()(int i, int j, int k) const {

      const IntVector c(i,j,k);

      var[c] = value;

    }
  };

  //convection
  template<typename FT, typename UT>
  struct ComputeConvection{
  };

  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT>{

  public:
    constCCVariable<double>& phi;
    constCCVariable<double>& rho;
    CCVariable<double>& rhs;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ){

      IntVector c(i,j,k);
      IntVector cm(i,j,k);
      IntVector cp(i,j,k);
      IntVector cmm(i,j,k);
      IntVector cpp(i,j,k);
      cm -= dir;
      cp -= dir;
      for (int i = 0; i < 3; i++){
        cmm[i] = cmm[i] - 2.*dir[i];
      }
      for (int i = 0; i < 3; i++){
        cpp[i] = cpp[i] + 2.*dir[i];
      }

      double Sup_up = phi[cm];
      double Sdn_up = phi[c];
      double r = ( phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] );

      double psi_up = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_up = std::max( 0.0, psi_up );

      double Sup_dn = phi[c];
      double Sdn_dn = phi[cm];
      r = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] );

      double psi_dn = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_dn = std::max( 0.0, psi_dn );

      double face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up );
      double face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn );

      double face_value_m = ( u[c] > 0.0 ) ? face_up : face_dn;

      Sup_up = phi[c];
      Sdn_up = phi[cp];
      r = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] );

      psi_up = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_up = std::max( 0.0, psi_up );

      Sup_dn = phi[cp];
      Sdn_dn = phi[c];
      r = ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] );

      psi_dn = std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) );
      psi_dn = std::max( 0.0, psi_dn );

      face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up );
      face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn );

      double face_value_p = ( u[cp] > 0.0 ) ? face_up : face_dn;

      //Done with interpolation, now compute conv and add to RHS:
      rhs[c] += A * ( 0.5 * ( rho[c] + rho[cp] ) * face_value_p * u[cp] -
                0.5 * ( rho[c] + rho[cm] ) * face_value_m * u[c] );

    }
  };

  template<typename UT>
  struct ComputeConvection<SFCXVariable<double>, UT>{

    constSFCXVariable<double>& phi;
    constCCVariable<double>& rho;
    SFCXVariable<double>& rhs;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( constSFCXVariable<double>& phi, SFCXVariable<double>& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ){}
  };

  template<typename UT>
  struct ComputeConvection<SFCYVariable<double>, UT>{
    constSFCYVariable<double>& phi;
    constCCVariable<double>& rho;
    SFCYVariable<double>& rhs;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( constSFCYVariable<double>& phi, SFCYVariable<double>& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ){}
  };

  template<typename UT>
  struct ComputeConvection<SFCZVariable<double>, UT>{
    constSFCZVariable<double>& phi;
    constCCVariable<double>& rho;
    SFCZVariable<double>& rhs;
    UT& u;
    IntVector dir;
    double A;

    ComputeConvection( constSFCZVariable<double>& phi, SFCZVariable<double>& rhs,
      constCCVariable<double>& rho, UT& u, IntVector dir, double A)
      : phi(phi), rhs(rhs), rho(rho), u(u), dir(dir), A(A){}

    void
    operator()(int i, int j, int k ){}
  };

  //discretization
  class DiscretizationTools{
  public:

  private:

  };
}
#endif
