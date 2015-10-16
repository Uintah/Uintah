#ifndef Uintah_Component_Arches_KFEUpdate_h
#define Uintah_Component_Arches_KFEUpdate_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

namespace Uintah{

  class KFEUpdate : public TaskInterface {

public:

    KFEUpdate( std::string task_name, int matl_index, std::vector<std::string> eqn_names, bool divide_out_density );
    ~KFEUpdate();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){}

    /** @brief Build instruction for this class **/
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::vector<std::string> eqn_names, bool divide_out_density ) :
        _task_name(task_name), _matl_index(matl_index), _eqn_names(eqn_names), _div_density(divide_out_density){}
      ~Builder(){}

      KFEUpdate* build()
      { return scinew KFEUpdate( _task_name, _matl_index, _eqn_names, _div_density ); }

      private:

      std::string _task_name;
      int _matl_index;
      std::vector<std::string> _eqn_names;
      bool _div_density;

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
    bool _div_density;


  };

  //Function definitions:

  KFEUpdate::KFEUpdate( std::string task_name, int matl_index, std::vector<std::string> eqn_names, bool divide_out_density ) :
  TaskInterface( task_name, matl_index ){

    _eqn_names = eqn_names;
    _div_density = divide_out_density;

  }

  KFEUpdate::~KFEUpdate()
  {
  }

  void KFEUpdate::problemSetup( ProblemSpecP& db ){

  }


  void KFEUpdate::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  }

  //This is the work for the task.  First, get the variables. Second, do the work!
  void KFEUpdate::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                SpatialOps::OperatorDatabase& opr ){
  }


  void KFEUpdate::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){

    //FUNCITON CALL     STRING NAME(VL)     DEPENDENCY    GHOST DW     VR
    //register_variable( "templated_variable", ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    typedef std::vector<std::string> SV;
    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){
      register_variable( *i, ArchesFieldContainer::MODIFIES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      std::string rhs_name = *i + "_RHS";
      register_variable( rhs_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    }
    register_variable( "density", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

  }

  namespace {

    struct TimeUpdateFunctor{

      CCVariable<double>& phi;
      constCCVariable<double>& rho;
      constCCVariable<double>& rhs;
      const double dt;
      const double V;

      TimeUpdateFunctor( CCVariable<double>& phi, constCCVariable<double>& rho,
        constCCVariable<double>& rhs, const double dt, const double V )
        : phi(phi), rho(rho), rhs(rhs), dt(dt), V(V){ }

      void
      operator()(int i, int j, int k){
        IntVector c(i,j,k);

        phi[c] = phi[c] + dt * rhs[c] / ( rho[c] * V );

      }
    };

  }

  void KFEUpdate::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                          SpatialOps::OperatorDatabase& opr ){

    constCCVariable<double>& rho =
      *(tsk_info->get_const_uintah_field<constCCVariable<double> >("density"));

    const double dt = tsk_info->get_dt();
    Vector DX = patch->dCell();
    const double V = DX.x()*DX.y()*DX.z();

    typedef std::vector<std::string> SV;

    for ( SV::iterator i = _eqn_names.begin(); i != _eqn_names.end(); i++){

      CCVariable<double>& phi = *(tsk_info->get_uintah_field<CCVariable<double> >(*i));
      constCCVariable<double>& rhs =
        *(tsk_info->get_const_uintah_field<constCCVariable<double> >(*i+"_RHS"));

      TimeUpdateFunctor time_update(phi, rho, rhs, dt, V);

    }
  }
}
#endif
