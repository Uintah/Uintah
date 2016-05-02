#ifndef Uintah_Component_Arches_KFEUpdate_h
#define Uintah_Component_Arches_KFEUpdate_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/DiscretizationTools.h>
#include <spatialops/util/TimeLogger.h>

namespace Uintah{

  template <typename T>
  class KFEUpdate : public TaskInterface {

public:

    KFEUpdate<T>( std::string task_name, int matl_index, std::vector<std::string> eqn_names );
    ~KFEUpdate<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){}

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::vector<std::string> eqn_names ) :
        _task_name(task_name), _matl_index(matl_index), _eqn_names(eqn_names){}
      ~Builder(){}

      KFEUpdate* build()
      { return scinew KFEUpdate( _task_name, _matl_index, _eqn_names ); }

      private:

      std::string _task_name;
      int _matl_index;
      std::vector<std::string> _eqn_names;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr ){};

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );

private:

    typedef typename VariableHelper<T>::ConstType CT;
    typedef typename VariableHelper<T>::XFaceType FXT;
    typedef typename VariableHelper<T>::YFaceType FYT;
    typedef typename VariableHelper<T>::ZFaceType FZT;
    typedef typename VariableHelper<T>::ConstXFaceType CFXT;
    typedef typename VariableHelper<T>::ConstYFaceType CFYT;
    typedef typename VariableHelper<T>::ConstZFaceType CFZT;

    std::vector<std::string> _eqn_names;


  };

  //Function definitions:
  template <typename T>
  KFEUpdate<T>::KFEUpdate( std::string task_name, int matl_index, std::vector<std::string> eqn_names ) :
  TaskInterface( task_name, matl_index ){

    _eqn_names = eqn_names;

  }

  template <typename T>
  KFEUpdate<T>::~KFEUpdate()
  {
  }

  template <typename T>
  void KFEUpdate<T>::problemSetup( ProblemSpecP& db ){

  }

  template <typename T>
  void KFEUpdate<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  }

  //This is the work for the task.  First, get the variables. Second, do the work!
  template <typename T>
  void KFEUpdate<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                SpatialOps::OperatorDatabase& opr ){
  }

  template <typename T>
  void KFEUpdate<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      std::string rhs_name = *i + "_rhs";
      register_variable( rhs_name, ArchesFieldContainer::MODIFIES, variable_registry, time_substep );
      register_variable( *i+"_x_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i+"_y_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i+"_z_flux", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }
  }

  template <typename T>
  void KFEUpdate<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

    const double dt = tsk_info->get_dt();
    Vector DX = patch->dCell();
    const double V = DX.x()*DX.y()*DX.z();

    typedef std::vector<std::string> SV;
    typedef typename VariableHelper<T>::ConstType CT;

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      T& phi = *(tsk_info->get_uintah_field<T>(*i));
      T& rhs = *(tsk_info->get_uintah_field<T>(*i+"_rhs"));
      CT& old_phi = *(tsk_info->get_const_uintah_field<CT>(*i));
      CFXT& x_flux = *(tsk_info->get_const_uintah_field<CFXT>(*i+"_x_flux"));
      CFYT& y_flux = *(tsk_info->get_const_uintah_field<CFYT>(*i+"_y_flux"));
      CFZT& z_flux = *(tsk_info->get_const_uintah_field<CFZT>(*i+"_z_flux"));

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());
      Vector Dx = patch->dCell();
      double ax = Dx.y() * Dx.z();
      double ay = Dx.z() * Dx.x();
      double az = Dx.x() * Dx.y();

#ifdef UINTAH_ENABLE_KOKKOS
      KokkosView3<double> k_phi = phi.getKokkosView();
      KokkosView3<const double> k_old_phi = old_phi.getKokkosView();
      KokkosView3<const double> k_rhs = rhs.getKokkosView();
      KokkosView3<const double> k_flux_x = x_flux.getKokkosView();
      KokkosView3<const double> k_flux_y = y_flux.getKokkosView();
      KokkosView3<const double> k_flux_z = z_flux.getKokkosView();

      //time update:
      Uintah::parallel_for( range, [&k_phi, &k_old_phi,
        &k_rhs, &k_flux_x, &k_flux_y, &k_flux_z,
        &ax, &ay, &az, &dt, &V](int i, int j, int k){

        //add in the convective term
        k_rhs(i,j,k) = k_rhs(i,j,k) - ( ax * ( k_flux_x(i+1,j,k) - k_flux_x(i,j,k) ) +
                                        ay * ( k_flux_y(i,j+1,k) - k_flux_y(i,j,k) ) +
                                        az * ( k_flux_z(i,j,k+1) - k_flux_z(i,j,k) ) );

        k_phi(i,j,k) = k_old_phi(i,j,k) + dt/V * k_rhs(i,j,k);

      });
#else

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("fe_update.out."+*i);
      timer.start("fe_update");
#endif
      //time update:
      Uintah::parallel_for( range, [&phi, &old_phi,
        &rhs, &x_flux, &y_flux, &z_flux,
        &ax, &ay, &az, &dt, &V](int i, int j, int k){

        //note: the source term should already be in RHS (if any) which is why we have a +=
        //add in the convective term
        rhs(i,j,k) = rhs(i,j,k) - ( ax * ( x_flux(i+1,j,k) - x_flux(i,j,k) ) +
                                    ay * ( y_flux(i,j+1,k) - y_flux(i,j,k) ) +
                                    az * ( z_flux(i,j,k+1) - z_flux(i,j,k) ) );

        phi(i,j,k) = old_phi(i,j,k) + dt/V * rhs(i,j,k);

      });
#ifdef DO_TIMINGS
      timer.stop("fe_update");
#endif
#endif

    }
  }
}
#endif
