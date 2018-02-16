#ifndef Uintah_Component_Arches_DynamicSmagorinskyHelper_h
#define Uintah_Component_Arches_DynamicSmagorinskyHelper_h

#include <CCA/Components/Arches/GridTools.h>
namespace Uintah {

  enum FILTER { THREEPOINTS, SIMPSON, BOX };

  static FILTER get_filter_from_string(std::string value){

    if ( value == "simpson" ){
      return SIMPSON;
    } else if ( value == "three_points" ){
      return THREEPOINTS;
    } else if ( value == "box" ){
      return BOX;
    } else {
      throw InvalidValue("Error: Filter type not recognized: "+value, __FILE__, __LINE__);
    }

  }

  template <typename V_T>
  struct FilterrhoVarT{
    FilterrhoVarT( V_T& i_var, Array3<double>& i_Fvar, constCCVariable<double>& i_rho, constCCVariable<double>& i_eps,
                int _i_n, int _j_n, int _k_n ,FILTER i_Type): 
      var(i_var), Fvar(i_Fvar), rho(i_rho), eps(i_eps), i_n(_i_n),
       j_n(_j_n),k_n(_k_n), Type(i_Type)
      {    
      
      if (Type == THREEPOINTS  ) {
      // Three points symmetric: eq. 2.49 : LES for compressible flows Garnier et al.
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              double my_value = abs(m) + abs(n) + abs(l)+3.0;
              w[m+1][n+1][l+1]= (1.0/std::pow(2.0,my_value)); 
            }
          }
        }
      wt = 1.;
      } else if (Type == SIMPSON) {
      // Simpson integration rule: eq. 2.50 : LES for compressible flows Garnier et al.
      // ref shows 1D case. For 3D case filter 3 times with 1D filter . 
      for ( int m = -1; m <= 1; m++ ){
        for ( int n = -1; n <= 1; n++ ){
          for ( int l = -1; l <= 1; l++ ){
            double my_value = -abs(m) - abs(n) - abs(l)+3.0;
            w[m+1][n+1][l+1] = std::pow(4.0,my_value); 
          }
        }
      }
        wt = std::pow(6.0,3.0);

      } else if (Type == BOX) {
      // Doing average on a box with three points
      for ( int m = -1; m <= 1; m++ ){
        for ( int n = -1; n <= 1; n++ ){
          for ( int l = -1; l <= 1; l++ ){
            w[m+1][n+1][l+1] = 1.0; 
          }
        }
      }
      wt = 27.;
    } else {
      throw InvalidValue("Error: Filter type not recognized. ", __FILE__, __LINE__);
    }
    }
    void operator()(int i, int j, int k) const {
        double F_var = 0.0; 
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              double vf = std::floor((eps(i+m,j+n,k+l)
                          + eps(i+m-i_n,j+n-j_n,k+l-k_n))/2.0);
              F_var += w[m+1][n+1][l+1]*(vf*var(i+m,j+n,k+l)*
                       (rho(i+m,j+n,k+l)+rho(i+m-i_n,j+n-j_n,k+l-k_n))/2.); 
                       //(rho(i,j,k)+rho(i+i_n,j+j_n,k+k_n))/2.); 
            }
          }
        }
        F_var /= wt;
        Fvar(i,j,k) = F_var;
    }


  private:
  
  V_T& var;
  Array3<double>& Fvar;
  constCCVariable<double>& eps;
  constCCVariable<double>& rho;
  int i_n;
  int j_n;
  int k_n;
  FILTER Type ;
  double w[3][3][3];
  double wt;
  };

  template <typename V_T>
  struct Filterdensity{
    Filterdensity( V_T& i_var, Array3<double>& i_Fvar, constCCVariable<double>& i_eps,
                int _i_n, int _j_n, int _k_n ,FILTER i_Type): 
      var(i_var), Fvar(i_Fvar), eps(i_eps), i_n(_i_n),
       j_n(_j_n),k_n(_k_n), Type(i_Type)
      {    
      
      if (Type == THREEPOINTS  ) {
      // Three points symmetric: eq. 2.49 : LES for compressible flows Garnier et al.
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              double my_value = abs(m) + abs(n) + abs(l)+3.0;
              w[m+1][n+1][l+1]= (1.0/std::pow(2.0,my_value)); 
            }
          }
        }
      wt = 1.;
      } else if (Type == SIMPSON) {
      // Simpson integration rule: eq. 2.50 : LES for compressible flows Garnier et al.
      // ref shows 1D case. For 3D case filter 3 times with 1D filter . 
      for ( int m = -1; m <= 1; m++ ){
        for ( int n = -1; n <= 1; n++ ){
          for ( int l = -1; l <= 1; l++ ){
            double my_value = -abs(m) - abs(n) - abs(l)+3.0;
            w[m+1][n+1][l+1] = std::pow(4.0,my_value); 
          }
        }
      }
        wt = std::pow(6.0,3.0);

      } else if (Type == BOX) {
      // Doing average on a box with three points
      for ( int m = -1; m <= 1; m++ ){
        for ( int n = -1; n <= 1; n++ ){
          for ( int l = -1; l <= 1; l++ ){
            w[m+1][n+1][l+1] = 1.0; 
          }
        }
      }
      wt = 27.;
    } else {
      throw InvalidValue("Error: Filter type not recognized. ", __FILE__, __LINE__);
    }
    }
    void operator()(int i, int j, int k) const {
        double F_var = 0.0; 
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              //double vf = std::floor((eps(i+m,j+n,k+l)
              //            + eps(i+m-i_n,j+n-j_n,k+l-k_n))/2.0);
              double vf = eps(i+m,j+n,k+l);
              F_var += w[m+1][n+1][l+1]* (vf*var(i+m,j+n,k+l)+(1.-vf)*var(i,j,k)); 
            }
          }
        }
        F_var /= wt;
        Fvar(i,j,k) = F_var;
    }


  private:
  
  V_T& var;
  Array3<double>& Fvar;
  constCCVariable<double>& eps;
  int i_n;
  int j_n;
  int k_n;
  FILTER Type ;
  double w[3][3][3];
  double wt;
  };

  template <typename V_T>
  struct FilterVarT{
    FilterVarT( V_T& i_var, Array3<double>& i_Fvar, constCCVariable<double>& i_eps,
                int _i_n, int _j_n, int _k_n ,FILTER i_Type): 
      var(i_var), Fvar(i_Fvar), eps(i_eps), i_n(_i_n),
       j_n(_j_n),k_n(_k_n), Type(i_Type)
      {    
      
      if (Type == THREEPOINTS  ) {
      // Three points symmetric: eq. 2.49 : LES for compressible flows Garnier et al.
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              double my_value = abs(m) + abs(n) + abs(l)+3.0;
              w[m+1][n+1][l+1]= (1.0/std::pow(2.0,my_value)); 
            }
          }
        }
      wt = 1.;
      } else if (Type == SIMPSON) {
      // Simpson integration rule: eq. 2.50 : LES for compressible flows Garnier et al.
      // ref shows 1D case. For 3D case filter 3 times with 1D filter . 
      for ( int m = -1; m <= 1; m++ ){
        for ( int n = -1; n <= 1; n++ ){
          for ( int l = -1; l <= 1; l++ ){
            double my_value = -abs(m) - abs(n) - abs(l)+3.0;
            w[m+1][n+1][l+1] = std::pow(4.0,my_value); 
          }
        }
      }
        wt = std::pow(6.0,3.0);

      } else if (Type == BOX) {
      // Doing average on a box with three points
      for ( int m = -1; m <= 1; m++ ){
        for ( int n = -1; n <= 1; n++ ){
          for ( int l = -1; l <= 1; l++ ){
            w[m+1][n+1][l+1] = 1.0; 
          }
        }
      }
      wt = 27.;
    } else {
      throw InvalidValue("Error: Filter type not recognized. ", __FILE__, __LINE__);
    }
    }
    void operator()(int i, int j, int k) const {
        double F_var = 0.0; 
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              //double vf = std::floor((eps(i+m,j+n,k+l)
              //            + eps(i+m-i_n,j+n-j_n,k+l-k_n))/2.0);
              double vf = eps(i+m,j+n,k+l);
              F_var += w[m+1][n+1][l+1]* vf*var(i+m,j+n,k+l); 
            }
          }
        }
        F_var /= wt;
        Fvar(i,j,k) = F_var;
    }


  private:
  
  V_T& var;
  Array3<double>& Fvar;
  constCCVariable<double>& eps;
  int i_n;
  int j_n;
  int k_n;
  FILTER Type ;
  double w[3][3][3];
  double wt;
  };


  struct computeIsInsij{
    computeIsInsij(Array3<double>& i_IsI, Array3<double>& i_s11, Array3<double>& i_s22, Array3<double>& i_s33, 
                   Array3<double>& i_s12, Array3<double>& i_s13, Array3<double>& i_s23,
                   const Array3<double>& i_uVel, const Array3<double>& i_vVel, const Array3<double>& i_wVel,
                   const Array3<double>& i_CCuVel, const Array3<double>& i_CCvVel, const Array3<double>& i_CCwVel, const Vector& i_Dx):
                   IsI(i_IsI), s11(i_s11), s22(i_s22), s33(i_s33), s12(i_s12), s13(i_s13), s23(i_s23),
                   uVel(i_uVel), vVel(i_vVel), wVel(i_wVel), CCuVel(i_CCuVel), CCvVel(i_CCvVel), CCwVel(i_CCwVel), Dx(i_Dx)
    {}

    void
    operator()(int i, int j, int k ) const {

  double uep = 0.0;
  double uwp = 0.0;

  double vep = 0.0;
  double vwp = 0.0;

  double wep = 0.0;
  double wwp = 0.0;

  double unp = 0.0;
  double usp = 0.0;

  double vnp = 0.0;
  double vsp = 0.0;

  double wnp = 0.0;
  double wsp = 0.0;

  double utp = 0.0;
  double ubp = 0.0;

  double vtp = 0.0;
  double vbp = 0.0;

  double wtp = 0.0;
  double wbp = 0.0;

  // x-dir 
  {
  STENCIL3_1D(0);
  uep = uVel(IJK_P_);
  uwp = uVel(IJK_);

  vep = 0.50 * CCvVel(IJK_P_);
  vwp = 0.50 * CCvVel(IJK_M_);

  wep = 0.50 * CCwVel(IJK_P_);
  wwp = 0.50 * CCwVel(IJK_M_);
  }

  // y-dir
  {
  STENCIL3_1D(1);
  unp = 0.50 * CCuVel(IJK_P_);
  usp = 0.50 * CCuVel(IJK_M_);

  vnp = vVel(IJK_P_);
  vsp = vVel(IJK_);

  wnp = 0.50 * CCwVel(IJK_P_);
  wsp = 0.50 * CCwVel(IJK_M_);
  }

  // z-dir
  {
  STENCIL3_1D(2);

  utp = 0.50 * CCuVel(IJK_P_);
  ubp = 0.50 * CCuVel(IJK_M_);

  vtp = 0.50 * CCvVel(IJK_P_);
  vbp = 0.50 * CCvVel(IJK_M_);

  wtp = wVel(IJK_P_);
  wbp = wVel(IJK_);
  }
  
  s11(IJK_) = (uep-uwp)/Dx.x();
  s22(IJK_) = (vnp-vsp)/Dx.y();
  s33(IJK_) = (wtp-wbp)/Dx.z();
  s12(IJK_) = 0.50 * ((unp-usp)/Dx.y() + (vep-vwp)/Dx.x());
  s13(IJK_) = 0.50 * ((utp-ubp)/Dx.z() + (wep-wwp)/Dx.x());
  s23(IJK_) = 0.50 * ((vtp-vbp)/Dx.z() + (wnp-wsp)/Dx.y());

  IsI(IJK_) = 2.0 * ( std::pow(s11(IJK_),2.0) + std::pow(s22(IJK_),2.0) + std::pow(s33(IJK_),2.0)
            + 2.0 * ( std::pow(s12(IJK_),2.0) + std::pow(s13(IJK_),2.0) + std::pow(s23(IJK_),2.0) ) ); 

  IsI(IJK_) = std::sqrt( IsI(IJK_) ); 
    }
  private:
 
  Array3<double>& IsI;
  Array3<double>& s11;
  Array3<double>& s22;
  Array3<double>& s33;
  Array3<double>& s12;
  Array3<double>& s13;
  Array3<double>& s23;
  const Array3<double>& uVel;
  const Array3<double>& vVel;
  const Array3<double>& wVel;
  const Array3<double>& CCuVel;
  const Array3<double>& CCvVel;
  const Array3<double>& CCwVel;
  const Vector& Dx;
  };

  struct computefilterIsInsij{
    computefilterIsInsij(Array3<double>& i_filterIsI, Array3<double>& i_filters11, Array3<double>& i_filters22, 
                         Array3<double>& i_filters33, Array3<double>& i_filters12, Array3<double>& i_filters13, 
                         Array3<double>& i_filters23,
                         constSFCXVariable<double> i_filterRhoU, constSFCYVariable<double> i_filterRhoV,
                         constSFCZVariable<double> i_filterRhoW, constCCVariable<double> i_filterRho,
                         const Vector& i_Dx):
                         filterIsI(i_filterIsI), filters11(i_filters11), filters22(i_filters22), 
                         filters33(i_filters33), filters12(i_filters12), filters13(i_filters13), 
                         filters23(i_filters23), filterRhoU(i_filterRhoU), filterRhoV(i_filterRhoV),
                         filterRhoW(i_filterRhoW),filterRho(i_filterRho), Dx(i_Dx)
  {}

  void
  operator()(int i, int j, int k ) const {


  const double fuep = filterRhoU(i+1,j,k) /
         (0.5 * (filterRho(i,j,k) + filterRho(i+1,j,k)));

  const double fuwp = filterRhoU(i,j,k)/
         (0.5 * (filterRho(i,j,k) + filterRho(i-1,j,k)));

  //note: we have removed the (1/2) from the denom. because
  //we are multiplying by (1/2) for Sij
  const double funp = ( 0.5 * filterRhoU(i+1,j+1,k) /
         ( (filterRho(i,j+1,k) + filterRho(i+1,j+1,k)))
         + 0.5 * filterRhoU(i,j+1,k) /
         ( (filterRho(i,j+1,k) + filterRho(i-1,j+1,k))));

  const double fusp = ( 0.5 * filterRhoU(i+1,j-1,k) /
         ( (filterRho(i,j-1,k) + filterRho(i+1,j-1,k)) )
         + 0.5 * filterRhoU(i,j-1,k) /
         ( (filterRho(i,j-1,k) + filterRho(i-1,j-1,k))));

  const double futp = ( 0.5 * filterRhoU(i+1,j,k+1) /
         ( (filterRho(i,j,k+1) + filterRho(i+1,j,k+1)) )
         + 0.5 * filterRhoU(i,j,k+1) /
         ( (filterRho(i,j,k+1) + filterRho(i-1,j,k+1))));

  const double fubp = ( 0.5 * filterRhoU(i+1,j,k-1) /
         ( ( filterRho(i,j,k-1) + filterRho(i+1,j,k-1)))
         + 0.5 * filterRhoU(i,j,k-1) /
         ( (filterRho(i,j,k-1) + filterRho(i-1,j,k-1))));

  const double fvnp = filterRhoV(i,j+1,k) /
         ( 0.5 * (filterRho(i,j,k) + filterRho(i,j+1,k)));

  const double fvsp = filterRhoV(i,j,k) /
         ( 0.5 * (filterRho(i,j,k) + filterRho(i,j-1,k)));

  const double fvep = ( 0.5 * filterRhoV(i+1,j+1,k)/
         ( (filterRho(i+1,j,k) +filterRho(i+1,j+1,k)))
         + 0.5 * filterRhoV(i+1,j,k)/
         ( (filterRho(i+1,j,k) + filterRho(i+1,j-1,k))));

  const double fvwp = ( 0.5 * filterRhoV(i-1,j+1,k)/
         ( (filterRho(i-1,j,k) + filterRho(i-1,j+1,k)))
         + 0.5 * filterRhoV(i-1,j,k)/
         ( (filterRho(i-1,j,k) + filterRho(i-1,j-1,k))));

  const double fvtp = ( 0.5 * filterRhoV(i,j+1,k+1) /
         ( (filterRho(i,j,k+1) + filterRho(i,j+1,k+1)))
         + 0.5 * filterRhoV(i,j,k+1) /
         ( (filterRho(i,j,k+1) + filterRho(i,j-1,k+1))));

  const double fvbp = ( 0.5 * filterRhoV(i,j+1,k-1)/
         ( (filterRho(i,j,k-1) + filterRho(i,j+1,k-1)))
         + 0.5 * filterRhoV(i,j,k-1) /
         ( (filterRho(i,j,k-1) + filterRho(i,j-1,k-1))));

  const double fwtp = filterRhoW(i,j,k+1) /
         ( 0.5 * (filterRho(i,j,k) + filterRho(i,j,k+1)));

  const double fwbp = filterRhoW(i,j,k) /
         ( 0.5 * (filterRho(i,j,k) + filterRho(i,j,k-1)));

  const double fwep = ( 0.5 * filterRhoW(i+1,j,k+1) /
         ( (filterRho(i+1,j,k) + filterRho(i+1,j,k+1)))
         + 0.5 * filterRhoW(i+1,j,k) /
         ( (filterRho(i+1,j,k) + filterRho(i+1,j,k-1))));

  const double fwwp = ( 0.5 * filterRhoW(i-1,j,k+1) /
         ( (filterRho(i-1,j,k) + filterRho(i-1,j,k+1)))
         + 0.5 * filterRhoW(i-1,j,k) /
         ( (filterRho(i-1,j,k) + filterRho(i-1,j,k-1))));

  const double fwnp = ( 0.5 * filterRhoW(i,j+1,k+1)/
         ( (filterRho(i,j+1,k) + filterRho(i,j+1,k+1)))
         + 0.5 * filterRhoW(i,j+1,k) /
         ( (filterRho(i,j+1,k) + filterRho(i,j+1,k-1))));

  const double fwsp = ( 0.5 * filterRhoW(i,j-1,k+1)/
         ( (filterRho(i,j-1,k) + filterRho(i,j-1,k+1)))
         + 0.5 * filterRhoW(i,j-1,k)/
             ( (filterRho(i,j-1,k) + filterRho(i,j-1,k-1))));

  //calculate the filtered strain rate tensor
  filters11(i,j,k) = (fuep-fuwp)/Dx.x();
  filters22(i,j,k) = (fvnp-fvsp)/Dx.y();
  filters33(i,j,k) = (fwtp-fwbp)/Dx.z();
  filters12(i,j,k) = 0.5*((funp-fusp)/Dx.y() + (fvep-fvwp)/Dx.x());
  filters13(i,j,k) = 0.5*((futp-fubp)/Dx.z() + (fwep-fwwp)/Dx.x());
  filters23(i,j,k) = 0.5*((fvtp-fvbp)/Dx.z() + (fwnp-fwsp)/Dx.y());
  filterIsI(i,j,k) = std::sqrt(2.0*(filters11(i,j,k)*filters11(i,j,k) 
                     + filters22(i,j,k)*filters22(i,j,k) + filters33(i,j,k)*filters33(i,j,k)+
                     2.0*(filters12(i,j,k)*filters12(i,j,k) + 
                      filters13(i,j,k)*filters13(i,j,k) + filters23(i,j,k)*filters23(i,j,k))));
  }
  private:
  Array3<double>& filterIsI;
  Array3<double>& filters11;
  Array3<double>& filters22;
  Array3<double>& filters33;
  Array3<double>& filters12;
  Array3<double>& filters13;
  Array3<double>& filters23;
  constSFCXVariable<double>& filterRhoU;
  constSFCYVariable<double>& filterRhoV;
  constSFCZVariable<double>& filterRhoW;
  constCCVariable<double>& filterRho;
  const Vector& Dx;
  };
} //namespace
#endif
