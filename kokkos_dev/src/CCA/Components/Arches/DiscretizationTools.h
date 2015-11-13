#ifndef Uintah_Component_Arches_DISCRETIZATIONTOOLS_h
#define Uintah_Component_Arches_DISCRETIZATIONTOOLS_h

namespace Uintah{

  enum DIR {XDIR, YDIR, ZDIR};
  enum LIMITER {SUPERBEE, ROE}; 

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

#define CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2) \
      double phi_e = ( phi[c] + phi[cp] )/ 2.; \
      double phi_w = ( phi[c] + phi[cm] )/ 2.; \
      double u_e = ( u[cu_e] + u[cu_e2] ) / 2.; \
      double u_w = ( u[cu_w] + u[cu_w2] ) / 2.; \
      rhs[c] += -A * ( phi_e * u_e - phi_w * u_w );

#define SUPERBEEMACRO(r) \
      psi = ( r < huge ) ? std::max( std::min( 2.*r, 1.0), std::min(r, 2.0 ) ) : 2.0; \
      psi = std::max( 0.0, psi_up ) \

#define ROEMACRO(r) \
      psi = ( r < huge ) ? std::min(r, 0.0) : 2.0; \
      psi = std::max(0.0, psi); 

#define FLUXLIM(c,cm,cp,cmm,cpp, LIMTYPE) \
      double psi; \
      double psi_up; \
      double psi_dn; \
      double Sup_up = phi[cm]; \
      double Sdn_up = phi[c]; \
      double r = (phi[cm] - phi[cmm] ) / ( phi[c] - phi[cm] ); \
      LIMTYPE(r); \
      psi_up = psi; \
      double Sup_dn = phi[c]; \
      double Sdn_dn = phi[cm]; \
      r = ( phi[cp] - phi[c] ) / ( phi[c] - phi[cm] ); \
      LIMTYPE(r); \
      psi_dn = psi; \
      double face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up ); \
      double face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn ); \
      double face_value_m = ( u[c] > 0.0 ) ? face_up : face_dn; \
      Sup_up = phi[c]; \
      Sdn_up = phi[cp]; \
      r = ( phi[c] - phi[cm] ) / ( phi[cp] - phi[c] ); \
      LIMTYPE(r); \
      psi_up = psi; \
      Sup_dn = phi[cp]; \
      Sdn_dn = phi[c];  \
      r =  ( phi[cpp] - phi[cp] ) / ( phi[cp] - phi[c] ); \
      LIMTYPE(r); \
      psi_dn = psi; \
      face_up = Sup_up + 0.5 * psi_up * ( Sdn_up - Sup_up ); \
      face_dn = Sup_dn + 0.5 * psi_dn * ( Sdn_dn - Sup_dn ); \
      double face_value_p = ( u[cp] > 0.0 ) ? face_up : face_dn; \
      rhs[c] += -A * ( face_value_p * u[cp] - face_value_m * u[c] ); 

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
  template<typename PT, typename UT, int MYDIR, int MYLIMITER>
  struct ComputeConvection{

    //THIS IS A GENERIC PLACEHOLDER THAT DOES NOTHING!
    // It is used for face centered template parameters + flux limiters, which aren't supported now. 

    typedef typename VariableHelper<PT >::ConstType ConstPT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;
    const double huge; 

    ComputeConvection( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {
    }
  };

  template<typename PT, typename UT, int MYDIR>
  struct ComputeConvectionCentral{
  };

  // CCVariable<double> 
  //
  //SUPERBEE
  // x-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, XDIR, SUPERBEE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    double A;
    const double huge; 

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      IDIR; 

      FLUXLIM(c,cm,cp,cmm,cpp, SUPERBEEMACRO); 

    }
  };
  // y-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, YDIR, SUPERBEE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    double A;
    const double huge; 

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      JDIR; 

      FLUXLIM(c,cm,cp,cmm,cpp,SUPERBEEMACRO); 

    }
  };
  // z-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, ZDIR, SUPERBEE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    double A;
    const double huge; 

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      KDIR; 

      FLUXLIM(c,cm,cp,cmm,cpp,SUPERBEEMACRO); 

    }
  };

  //ROE MINMOD
  //
  // x-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, XDIR, ROE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    double A;
    const double huge; 

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      IDIR; 

      FLUXLIM(c,cm,cp,cmm,cpp, ROEMACRO); 

    }
  };
  // y-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, YDIR, ROE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    double A;
    const double huge; 

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      JDIR; 

      FLUXLIM(c,cm,cp,cmm,cpp,ROEMACRO); 

    }
  };
  // z-dir
  template<typename UT>
  struct ComputeConvection<CCVariable<double>, UT, ZDIR, ROE>{

  public:
    constCCVariable<double>& phi;
    CCVariable<double>& rhs;
    UT& u;
    double A;
    const double huge; 

    ComputeConvection( constCCVariable<double>& phi, CCVariable<double>& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A), huge(1e10){}

    void
    operator()(int i, int j, int k ) const {

      KDIR; 

      FLUXLIM(c,cm,cp,cmm,cpp,ROEMACRO); 

    }
  };

  // CCVariable<double> 
  //
  // x-dir
  template<typename UT>
  struct ComputeConvectionCentral<CCVariable<double>, UT, XDIR>{

    typedef typename VariableHelper<CCVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<CCVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{
    }
  };
  // y-dir
  template<typename UT>
  struct ComputeConvectionCentral<CCVariable<double>, UT, YDIR>{

    typedef typename VariableHelper<CCVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<CCVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{
    }
  };
  // y-dir
  template<typename UT>
  struct ComputeConvectionCentral<CCVariable<double>, UT, ZDIR>{

    typedef typename VariableHelper<CCVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<CCVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{
    }
  };

  // SFCXVariable<double> 
  //
  // x-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCXVariable<double>, UT, XDIR>{

    typedef typename VariableHelper<SFCXVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCXVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      XIDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // y-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCXVariable<double>, UT, YDIR>{

    typedef typename VariableHelper<SFCXVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCXVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YIDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // z-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCXVariable<double>, UT, ZDIR>{

    typedef typename VariableHelper<SFCXVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCXVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZIDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // SFCYVariable<double> 
  //
  // x-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCYVariable<double>, UT, XDIR>{

    typedef typename VariableHelper<SFCYVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCYVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YIDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // y-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCYVariable<double>, UT, YDIR>{

    typedef typename VariableHelper<SFCYVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCYVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YJDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // z-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCYVariable<double>, UT, ZDIR>{

    typedef typename VariableHelper<SFCYVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCYVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      YKDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // SFCZVariable<double> 
  //
  // x-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCZVariable<double>, UT, XDIR>{

    typedef typename VariableHelper<SFCZVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCZVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZIDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // y-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCZVariable<double>, UT, YDIR>{

    typedef typename VariableHelper<SFCZVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCZVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZJDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

    }
  };

  // z-dir
  template<typename UT>
  struct ComputeConvectionCentral<SFCZVariable<double>, UT, ZDIR>{

    typedef typename VariableHelper<SFCZVariable<double> >::ConstType ConstPT;
    typedef typename VariableHelper<SFCZVariable<double> >::Type PT;

    ConstPT& phi;
    PT& rhs;
    UT& u;
    double A;

    ComputeConvectionCentral( ConstPT& phi, PT& rhs,
      UT& u, double A)
      : phi(phi), rhs(rhs), u(u), A(A){}

    void
    operator()(int i, int j, int k ) const{

      ZKDIR; 

      CENTRAL(c,cm,cp,cu_w,cu_w2,cu_e,cu_e2);  

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
