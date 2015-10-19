#ifndef Uintah_Component_Arches_KFEUpdate_h
#define Uintah_Component_Arches_KFEUpdate_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/DiscretizationTools.h>

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

    //FUNCITON CALL     STRING NAME(VL)     DEPENDENCY    GHOST DW     VR
    //register_variable( "templated_variable", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      std::string rhs_name = *i + "_RHS";
      register_variable( rhs_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }

  }

  namespace {

    template <typename T>
    struct TimeUpdateFunctor{

      typedef typename VariableHelper<T>::ConstType CT;
      T& phi;
      CT& rhs;
      const double dt;
      const double V;

      TimeUpdateFunctor( T& phi,
        CT& rhs, const double dt, const double V )
        : phi(phi), rhs(rhs), dt(dt), V(V){ }

      void
      operator()(int i, int j, int k) const{

        IntVector c(i,j,k);
        phi[c] = phi[c] + dt/V * rhs[c];

      }
    };

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
      CT& rhs =
        *(tsk_info->get_const_uintah_field<CT>(*i+"_RHS"));

      TimeUpdateFunctor<T> time_update(phi, rhs, dt, V);

    }
  }
}
#endif
