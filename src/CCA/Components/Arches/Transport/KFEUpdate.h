#ifndef Uintah_Component_Arches_KFEUpdate_h
#define Uintah_Component_Arches_KFEUpdate_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/Directives.h>
#include <spatialops/util/TimeLogger.h>

namespace Uintah{

  template <typename T>
  class KFEUpdate : public TaskInterface {

public:

    KFEUpdate<T>( std::string task_name, int matl_index );
    ~KFEUpdate<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){}

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        _task_name(task_name), _matl_index(matl_index) {}
      ~Builder(){}

      KFEUpdate* build()
      { return scinew KFEUpdate( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

private:

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    typedef typename ArchesCore::VariableHelper<T>::XFaceType FXT;
    typedef typename ArchesCore::VariableHelper<T>::YFaceType FYT;
    typedef typename ArchesCore::VariableHelper<T>::ZFaceType FZT;
    typedef typename ArchesCore::VariableHelper<T>::ConstXFaceType CFXT;
    typedef typename ArchesCore::VariableHelper<T>::ConstYFaceType CFYT;
    typedef typename ArchesCore::VariableHelper<T>::ConstZFaceType CFZT;

    std::vector<std::string> _eqn_names;


  };

  //Function definitions:
  template <typename T>
  KFEUpdate<T>::KFEUpdate( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){}

  template <typename T>
  KFEUpdate<T>::~KFEUpdate()
  {
  }

  template <typename T>
  void KFEUpdate<T>::problemSetup( ProblemSpecP& db ){
    for (ProblemSpecP eqn_db = db->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){
      std::string scalar_name;

      eqn_db->getAttribute("label", scalar_name);
      _eqn_names.push_back(scalar_name);

    }
  }

  template <typename T>
  void KFEUpdate<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  }

  //This is the work for the task.  First, get the variables. Second, do the work!
  template <typename T>
  void KFEUpdate<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

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
  void KFEUpdate<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    const double dt = tsk_info->get_dt();
    Vector DX = patch->dCell();
    const double V = DX.x()*DX.y()*DX.z();

    typedef std::vector<std::string> SV;
    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

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

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("kokkos_fe_update.out."+*i);
      timer.start("work");
#endif
      //time update:
      Uintah::parallel_for( range, [&](int i, int j, int k){

        //note: the source term should already be in RHS (if any) which is why we have a +=
        //add in the convective term
        rhs(i,j,k) = rhs(i,j,k) - ( ax * ( x_flux(i+1,j,k) - x_flux(i,j,k) ) +
                                    ay * ( y_flux(i,j+1,k) - y_flux(i,j,k) ) +
                                    az * ( z_flux(i,j,k+1) - z_flux(i,j,k) ) );
        phi(i,j,k) = old_phi(i,j,k) + dt/V * rhs(i,j,k);

      });
#ifdef DO_TIMINGS
      timer.stop("work");
#endif

    }
  }
}
#endif
