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
  struct FilterVarT{
    FilterVarT( V_T& i_var, Array3<double>& i_Fvar, FILTER i_Type): 
      var(i_var), Fvar(i_Fvar), Type(i_Type)
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
      throw InvalidValue("Error: Filter type not recognized: "+Type, __FILE__, __LINE__);
    }
    }
    void operator()(int i, int j, int k) const {
        double F_var = 0.0; 
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              F_var += w[m+1][n+1][l+1]* var(i+m,j+n,k+l); 
            }
          }
        }
        F_var /= wt;
        Fvar(i,j,k) = F_var;
    }


  private:
  
  V_T& var;
  Array3<double>& Fvar;
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


} //namespace
#endif
