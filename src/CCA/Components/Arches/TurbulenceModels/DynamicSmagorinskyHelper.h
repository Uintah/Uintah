#ifndef Uintah_Component_Arches_DynamicSmagorinskyHelper_h
#define Uintah_Component_Arches_DynamicSmagorinskyHelper_h

#include <CCA/Components/Arches/GridTools.h>
namespace Uintah { namespace ArchesCore {

  enum FILTER { THREEPOINTS, SIMPSON, BOX };

  FILTER get_filter_from_string( const std::string & value );

  struct BCFilter {
	template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
    void apply_BC_filter_rho( const Patch* patch, grid_T& var,
    		                  grid_T& rho, grid_CT& vol_fraction,
	                          ExecutionObject<ExecSpace, MemSpace>& execObj){

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;


      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        CellIterator iter=patch->getFaceIterator(face, MEC);
        BlockRange range(iter.begin(), iter.end());

        parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
              var(i, j, k) = rho(i, j, k) ;
        });
      }


    }

    template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
    void apply_BC_rho( const Patch* patch, grid_T& var,
             		        grid_CT& rho,
 			                grid_CT& vol_fraction,
                            ExecutionObject<ExecSpace, MemSpace>& execObj){


      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);
        int fx = f_dir[0], fy = f_dir[1], fz = f_dir[2];
        CellIterator iter=patch->getFaceIterator(face, MEC);
        BlockRange range(iter.begin(), iter.end());

        parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
          var(i, j, k) = vol_fraction(i, j, k)*0.5*(rho(i, j, k)+rho(i-fx, j-fy, k-fz))+(1.-vol_fraction(i, j, k))*rho(i-fx, j-fy, k-fz);
        });
      }
    }

    template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
    void apply_zero_neumann( ExecutionObject<ExecSpace, MemSpace>& execObj, const Patch* patch, grid_T& var,
                             grid_CT& vol_fraction ){

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      IntVector dcdp=(patch->getCellHighIndex()-patch->getCellLowIndex()); // delta cells / delta patch

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        // FACE EXTRA CELLS ONLY, not edge cells, not corner cells  (invalid for faces that don't have extra cells)
        int pom=f_dir[0]+f_dir[1]+f_dir[2]; // plus or minus
        IntVector cutEdges=IntVector(pom,pom,pom)-f_dir;
        IntVector loVal  = (pom < 0) ?patch->getCellLowIndex()  + f_dir: patch->getCellLowIndex()  + f_dir*dcdp ; 
        IntVector HiVal  = (pom < 0) ?patch->getCellHighIndex() + f_dir*dcdp: patch->getCellHighIndex() +f_dir; 

        parallel_for(execObj, BlockRange(loVal,HiVal), KOKKOS_LAMBDA (int i, int j, int k){

          int im=i-f_dir[0];
          int jm=j-f_dir[1];
          int km=k-f_dir[2];

          if ( vol_fraction(i,j,k) > 1e-10 ){ // This loop only executes over extra cells
            var(i,j,k) = var(im,jm,km);
          }
        });
      }
    }

    //Need to pass extra HostV_T (type on host)  to be used by VariableHelper. VariableHelper can not use KokkosView3
    template <typename HostV_T, typename T, typename CT, typename grid_CT, typename ExecSpace, typename MemSpace>
    void apply_BC_rhou( const Patch* patch, T& var, CT& vel,
    		            grid_CT& rho, grid_CT& vol_fraction,
						ExecutionObject<ExecSpace, MemSpace>& execObj){

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

      ArchesCore::VariableHelper<HostV_T> var_help;
      IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        const double dot = vDir[0]*f_dir[0] + vDir[1]*f_dir[1] + vDir[2]*f_dir[2];

        //The face normal and the velocity are in parallel
        if (dot == -1) {
            //Face +
          CellIterator iter=patch->getFaceIterator(face, MEC);
          BlockRange range(iter.begin(), iter.end());

          Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
            int im=i-f_dir[0];
            int jm=j-f_dir[1];
            int km=k-f_dir[2];

            if ( vol_fraction(i,j,k) > 1e-10 ){
              var(im,jm,km) = vel(im,jm,km)*(rho(im,jm,km)+rho(i,j,k))/2.;
              var(i,j,k) = vel(im,jm,km);
            }
          });
        } else {
              // Face -
          CellIterator iter=patch->getFaceIterator(face, MEC);
          BlockRange range(iter.begin(), iter.end());

          Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
            int im=i-f_dir[0];
            int jm=j-f_dir[1];
            int km=k-f_dir[2];

            if ( vol_fraction(i,j,k) > 1e-10 ){
              var(i,j,k) = vel(i,j,k)*(rho(im,jm,km)+rho(i,j,k))/2.;
            }
          });
        }
      }
    }

    //Not use. Not tested after porting.
    template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT, typename VarHelper>
    void apply_zero_neumann(ExecutionObject<ExecSpace, MemSpace>& execObj, const Patch* patch, grid_T& var,
                             grid_CT& vol_fraction, VarHelper& var_help  ){ // NOT USED??

      std::vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

      IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

      for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        IntVector f_dir = patch->getFaceDirection(face);

        const double dot = vDir[0]*f_dir[0] + vDir[1]*f_dir[1] + vDir[2]*f_dir[2];

        //The face normal and the velocity are in parallel
        if (dot == -1) {
            //Face +
          CellIterator iter=patch->getFaceIterator(face, MEC);
          BlockRange range(iter.begin(), iter.end());

          Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){

            if ( vol_fraction(i,j,k) > 1e-10 ){
              int im=i-f_dir[0];
              int jm=j-f_dir[1];
              int km=k-f_dir[2];
              int imm=im-f_dir[0];
              int jmm=jm-f_dir[1];
              int kmm=km-f_dir[2];

              var(im,jm,km) = var(imm,jmm,kmm);
              var(i,j,k) = var(im,jm,km);
            }
          });
          } else {
              // Face -
          CellIterator iter=patch->getFaceIterator(face, MEC);
          BlockRange range(iter.begin(), iter.end());

          Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){

            if ( vol_fraction(i,j,k) > 1e-10 ){
              int im=i-f_dir[0];
              int jm=j-f_dir[1];
              int km=k-f_dir[2];

              var(i,j,k) = var(im,jm,km);
            }
          });
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
  //Need to pass extra HostV_T (type on host)  to be used by VariableHelper. VariableHelper can not use KokkosView3
  template <typename HostV_T, typename V_T, typename grid_T, typename grid_CT, typename ExecSpace, typename MemSpace>
  void applyFilter(V_T& var, grid_T& Fvar, grid_CT& rho,
		           grid_CT& eps, BlockRange range, ExecutionObject<ExecSpace, MemSpace>& execObj)
  {
  ArchesCore::VariableHelper<HostV_T> helper;
  const int i_n = helper.ioff;
  const int j_n = helper.joff;
  const int k_n = helper.koff;
  localW wl(w);
  const double wtl = wt;

  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
    double F_var = 0.0;
    for ( int m = -1; m <= 1; m++ ){
      for ( int n = -1; n <= 1; n++ ){
        for ( int l = -1; l <= 1; l++ ){
          double vf = std::floor((eps(i+m,j+n,k+l)
                      + eps(i+m-i_n,j+n-j_n,k+l-k_n))/2.0);
          F_var += wl.w[m+1][n+1][l+1]*(vf*var(i+m,j+n,k+l)*
                   (rho(i+m,j+n,k+l)+rho(i+m-i_n,j+n-j_n,k+l-k_n))/2.);
        }
      }
    }
    F_var /= wtl;
    F_var *= (eps(i,j,k)*eps(i-i_n,j-j_n,k-k_n));
    Fvar(i,j,k) = F_var;
  });
  }
  //  This filter does not weight the intrusion cells instead c value is used.
  //  used in density
  template <typename T, typename grid_T, typename grid_CT, typename ExecSpace, typename MemSpace>
  void applyFilter(T& var, grid_T& Fvar,
                   BlockRange range, grid_CT& eps, ExecutionObject<ExecSpace, MemSpace>& execObj )
  {

  localW wl(w);
  const double wtl = wt;
  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
    double F_var = 0.0;
    for ( int m = -1; m <= 1; m++ ){
      for ( int n = -1; n <= 1; n++ ){
        for ( int l = -1; l <= 1; l++ ){
          F_var += wl.w[m+1][n+1][l+1]*(eps(i+m,j+n,k+l)*var(i+m,j+n,k+l)
                 +(1.-eps(i+m,j+n,k+l))*var(i,j,k));
        }
      }
    }
    F_var /= wtl;
    Fvar(i,j,k) = F_var;
  });
  }
  // scalar filter
  template <typename V_T, typename grid_T, typename grid_CT, typename ExecSpace, typename MemSpace>
  void applyFilter(V_T& var, grid_T& Fvar,
		           grid_CT& eps, BlockRange range, ExecutionObject<ExecSpace, MemSpace>& execObj)
  {

  localW wl(w);
  const double wtl = wt;
  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
    double F_var = 0.0;
    for ( int m = -1; m <= 1; m++ ){
      for ( int n = -1; n <= 1; n++ ){
        for ( int l = -1; l <= 1; l++ ){
          F_var += wl.w[m+1][n+1][l+1]* eps(i+m,j+n,k+l)*var(i+m,j+n,k+l);
        }
      }
    }
    F_var /= wtl;
    Fvar(i,j,k) = F_var;
  });
  }

  struct localW{
	localW(double  a[][3][3]){
      for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
   	      for(int k=0; k<3; k++)
            w[i][j][k] = a[i][j][k];
	}
    double w[3][3][3];
  };

  private:

  FILTER Type ;
  double w[3][3][3];
  double wt;
  };

}} //namespace Uintah::ArchesCore
#endif
