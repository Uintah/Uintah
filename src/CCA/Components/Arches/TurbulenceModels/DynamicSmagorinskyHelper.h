#ifndef Uintah_Component_Arches_DynamicSmagorinskyHelper_h
#define Uintah_Component_Arches_DynamicSmagorinskyHelper_h

#include <CCA/Components/Arches/GridTools.h>
namespace Uintah { namespace ArchesCore {

  enum FILTER { THREEPOINTS, SIMPSON, BOX };

  FILTER get_filter_from_string( const std::string & value );

  struct BCFilter {
    void apply_BC_filter_rho( const Patch* patch, CCVariable<double>& var,
                            CCVariable<double>& rho, constCCVariable<double>& vol_fraction){

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;


      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
          IntVector c = *iter;

              var[c] = rho[c] ;
        }
      }


    }
    void apply_BC_rho( const Patch* patch, CCVariable<double>& var,
                            constCCVariable<double>& rho,
                            constCCVariable<double>& vol_fraction){


      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
          IntVector c = *iter;
          var[c] = vol_fraction[c]*0.5*(rho[c]+rho[c-f_dir])+(1.-vol_fraction[c])*rho[c-f_dir];
        }
      }
    }

    void apply_zero_neumann( const Patch* patch, CCVariable<double>& var,
                             constCCVariable<double>& vol_fraction ){

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
          IntVector c = *iter;

          if ( vol_fraction[c] > 1e-10 ){
            var[c] = var[c-f_dir];
          }
        }
      }
    }
    template <typename T, typename CT>
    void apply_BC_rhou( const Patch* patch, T& var, CT& vel,
                        constCCVariable<double>& rho, constCCVariable<double>& vol_fraction  ){

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

      ArchesCore::VariableHelper<T> var_help;
      IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        const double dot = vDir[0]*f_dir[0] + vDir[1]*f_dir[1] + vDir[2]*f_dir[2];

        //The face normal and the velocity are in parallel
        if (dot == -1) {
            //Face +
          for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
            IntVector c = *iter;

            if ( vol_fraction[c] > 1e-10 ){
              var[c-f_dir] = vel[c-f_dir]*(rho[c-f_dir]+rho[c])/2.;
              var[c] = vel[c-f_dir];
            }
          }
          } else {
              // Face -
          for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
            IntVector c = *iter;

            if ( vol_fraction[c] > 1e-10 ){
              var[c] = vel[c]*(rho[c-f_dir]+rho[c])/2.;
            }
          }
        }
      }
    }

    template <typename T>
    void apply_zero_neumann( const Patch* patch, T& var,
                             constCCVariable<double>& vol_fraction  ){

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

      ArchesCore::VariableHelper<T> var_help;
      IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        const double dot = vDir[0]*f_dir[0] + vDir[1]*f_dir[1] + vDir[2]*f_dir[2];

        //The face normal and the velocity are in parallel
        if (dot == -1) {
            //Face +
          for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
            IntVector c = *iter;

            if ( vol_fraction[c] > 1e-10 ){
              var[c-f_dir] = var[c-f_dir-f_dir];
              var[c] = var[c-f_dir];
            }
          }
          } else {
              // Face -
          for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
            IntVector c = *iter;

            if ( vol_fraction[c] > 1e-10 ){
              var[c] = var[c-f_dir];
            }
          }
        }
      }
    }
  };

  //----------------------------------------------------------------------------------------------------

  struct TestFilter {

    void get_w(FILTER Type)
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
  // rh*u filter
  template <typename V_T>
  void applyFilter(V_T& var, Array3<double>& Fvar, constCCVariable<double>& rho,
        constCCVariable<double>& eps, BlockRange range)
  {
  ArchesCore::VariableHelper<V_T> helper;
  const int i_n = helper.ioff;
  const int j_n = helper.joff;
  const int k_n = helper.koff;

  Uintah::parallel_for( range, [&](int i, int j, int k){
    double F_var = 0.0;
    for ( int m = -1; m <= 1; m++ ){
      for ( int n = -1; n <= 1; n++ ){
        for ( int l = -1; l <= 1; l++ ){
          double vf = std::floor((eps(i+m,j+n,k+l)
                      + eps(i+m-i_n,j+n-j_n,k+l-k_n))/2.0);
          F_var += w[m+1][n+1][l+1]*(vf*var(i+m,j+n,k+l)*
                   (rho(i+m,j+n,k+l)+rho(i+m-i_n,j+n-j_n,k+l-k_n))/2.);
        }
      }
    }
    F_var /= wt;
    F_var *= (eps(i,j,k)*eps(i-i_n,j-j_n,k-k_n));
    Fvar(i,j,k) = F_var;
  });
  }
  //  This filter does not weight the intrusion cells instead c value is used.
  //  used in density
  template <typename T>
  void applyFilter(T& var, Array3<double>& Fvar,
                  BlockRange range, constCCVariable<double>& eps )
  {

  Uintah::parallel_for( range, [&](int i, int j, int k){
    double F_var = 0.0;
    for ( int m = -1; m <= 1; m++ ){
      for ( int n = -1; n <= 1; n++ ){
        for ( int l = -1; l <= 1; l++ ){
          F_var += w[m+1][n+1][l+1]*(eps(i+m,j+n,k+l)*var(i+m,j+n,k+l)
                 +(1.-eps(i+m,j+n,k+l))*var(i,j,k));
        }
      }
    }
    F_var /= wt;
    Fvar(i,j,k) = F_var;
  });
  }
  // scalar filter
  template <typename V_T>
  void applyFilter(V_T& var, Array3<double>& Fvar,
        constCCVariable<double>& eps, BlockRange range)
  {

  Uintah::parallel_for( range, [&](int i, int j, int k){
    double F_var = 0.0;
    for ( int m = -1; m <= 1; m++ ){
      for ( int n = -1; n <= 1; n++ ){
        for ( int l = -1; l <= 1; l++ ){
          F_var += w[m+1][n+1][l+1]* eps(i+m,j+n,k+l)*var(i+m,j+n,k+l);
        }
      }
    }
    F_var /= wt;
    Fvar(i,j,k) = F_var;
  });
  }
  private:

  FILTER Type ;
  double w[3][3][3];
  double wt;
  };

}} //namespace Uintah::ArchesCore
#endif
