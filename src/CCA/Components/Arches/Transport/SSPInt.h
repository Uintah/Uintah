#ifndef Uintah_Component_Arches_SSPInt_h
#define Uintah_Component_Arches_SSPInt_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

namespace Uintah{

  template <typename T>
  class SSPInt : public TaskInterface {

public:

    SSPInt<T>( std::string task_name, int matl_index, std::vector<std::string> eqn_names );
    ~SSPInt<T>();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){}

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::vector<std::string> eqn_names ) :
        _task_name(task_name), _matl_index(matl_index), _eqn_names(eqn_names){}
      ~Builder(){}

      SSPInt* build()
      { return new SSPInt<T>( _task_name, _matl_index, _eqn_names ); }

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
    Vector _ssp_beta, _ssp_alpha, _time_factor;
    int _time_order;

  };

  //Function definitions:

  template <typename T>
  SSPInt<T>::SSPInt( std::string task_name, int matl_index, std::vector<std::string> eqn_names ) :
  TaskInterface( task_name, matl_index ){

    _eqn_names = eqn_names;

  }

  template <typename T>
  SSPInt<T>::~SSPInt()
  {
  }

  template <typename T>
  void SSPInt<T>::problemSetup( ProblemSpecP& db ){

    db->findBlock("TimeIntegrator")->getAttribute("order", _time_order);

    if ( _time_order == 1 ){

      _ssp_alpha[0] = 0.0;
      _ssp_alpha[1] = 0.0;
      _ssp_alpha[2] = 0.0;

      _ssp_beta[0]  = 1.0;
      _ssp_beta[1]  = 0.0;
      _ssp_beta[2]  = 0.0;

      _time_factor[0] = 1.0;
      _time_factor[1] = 0.0;
      _time_factor[2] = 0.0;

    } else if ( _time_order == 2 ) {

      _ssp_alpha[0]= 0.0;
      _ssp_alpha[1]= 0.5;
      _ssp_alpha[2]= 0.0;

      _ssp_beta[0]  = 1.0;
      _ssp_beta[1]  = 0.5;
      _ssp_beta[2]  = 0.0;

      _time_factor[0] = 1.0;
      _time_factor[1] = 1.0;
      _time_factor[2] = 0.0;

    } else if ( _time_order == 3 ) {

      _ssp_alpha[0] = 0.0;
      _ssp_alpha[1] = 0.75;
      _ssp_alpha[2] = 1.0/3.0;

      _ssp_beta[0]  = 1.0;
      _ssp_beta[1]  = 0.25;
      _ssp_beta[2]  = 2.0/3.0;

      _time_factor[0] = 1.0;
      _time_factor[1] = 0.5;
      _time_factor[2] = 1.0;

    } else {
      throw InvalidValue("Error: <TimeIntegrator> must have value: 1, 2, or 3 (representing the order).",__FILE__, __LINE__);
    }

  }


  template <typename T>
  void SSPInt<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

  }

  //This is the work for the task.  First, get the variables. Second, do the work!
  template <typename T>
  void SSPInt<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                              SpatialOps::OperatorDatabase& opr ){
  }


  template <typename T>
  void SSPInt<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    //FUNCITON CALL     STRING NAME(VL)     DEPENDENCY    GHOST DW     VR
    //register_variable( "templated_variable", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
    }

    register_variable( "density", ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW,  variable_registry, time_substep );
    register_variable( "density", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,  variable_registry, time_substep );

  }

  template <typename T>
  void SSPInt<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr ){

    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SVolField   SVolF;
    typedef SpatialOps::SpatFldPtr<SVolF> SVolFP;
    typedef SpatialOps::SpatFldPtr<T> SFTP;

    typedef std::vector<std::string> SV;

    const int time_substep = tsk_info->get_time_substep();

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      //we don't actually modify density here, but need to trick the system
      //for now because density is appearing twice in the const fields and
      //we only allow it to appear once.
      //Perhaps the better thing to do is to create a new variable called
      //old_density so that we can get it as const
      SFTP phi   = tsk_info->get_so_field<T>( *i );
      SVolFP rho = tsk_info->get_so_field<T>( "density" );

      SFTP   const old_phi = tsk_info->get_const_so_field<T>( *i );
      SVolFP const old_rho = tsk_info->get_const_so_field<T>( "density" );

      double alpha = _ssp_alpha[time_substep];
      double beta  = _ssp_beta[time_substep];

      //Weighting:
      *phi <<= ( alpha * ( *old_rho * *old_phi) + beta * ( *rho * *phi) ) / ( alpha * *old_rho + beta * *rho );

    }
  }
}
#endif
