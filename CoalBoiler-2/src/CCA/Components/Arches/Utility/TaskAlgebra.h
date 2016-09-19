#ifndef Uintah_Component_Arches_TaskAlgebra_h
#define Uintah_Component_Arches_TaskAlgebra_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename T>
  class TaskAlgebra : public TaskInterface {

public:

    enum EXPR {ADD, SUBTRACT, MULTIPLY, DIVIDE, DIVIDE_CONST_VARIABLE,
               DIVIDE_VARIABLE_CONST, POW, EXP};

    TaskAlgebra<T>( std::string task_name, int matl_index );
    ~TaskAlgebra<T>();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (TaskAlgebra) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      TaskAlgebra* build()
      { return scinew TaskAlgebra<T>( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

protected:

    void register_initialize(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep );

    void register_compute_bcs(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

  typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

  struct Operation{

    bool create_new_variable;
    bool use_constant;

    std::string dep;
    std::string ind1;
    std::string ind2;

    double constant;

    EXPR expression_type;

  };

  typedef std::map<std::string, Operation> OPMAP;
  typedef std::vector<std::string> STRVEC;

  OPMAP all_operations;
  STRVEC order;

  /** @brief Search a vector for a string **/
  inline bool use_variable( std::string item, std::vector<std::string> vector){
    return std::find(vector.begin(), vector.end(), item)!=vector.end();
  }

  };

  // Class Implmentations --------------------------------------------------------------------------
  template <typename T>
  TaskAlgebra<T>::TaskAlgebra( std::string task_name, int matl_index) :
  TaskInterface( task_name, matl_index )
  {}

  template <typename T>
  TaskAlgebra<T>::~TaskAlgebra()
  {}

  // Problem Setup ---------------------------------------------------------------------------------
  template <typename T>
  void TaskAlgebra<T>::problemSetup( ProblemSpecP& db ){

    for ( ProblemSpecP op_db=db->findBlock("op"); op_db != 0; op_db=op_db->findNextBlock("op") ){

      Operation new_op;

      //get the name of this op
      std::string label;
      op_db->getAttribute("label",label);

      //does it create a new variable?
      new_op.create_new_variable = false;
      if ( op_db->findBlock("new_variable") ){
        new_op.create_new_variable = true;
      }

      //get variable names:
      op_db->require("dep", new_op.dep);
      op_db->require("ind1", new_op.ind1);

      //get the algebriac expression
      std::string value;
      op_db->getAttribute( "type", value );

      new_op.use_constant = false;
      if ( op_db->findBlock("ind2")){
        op_db->require("ind2", new_op.ind2);
      } else if ( op_db->findBlock("constant") ){
        op_db->require("constant", new_op.constant);
        new_op.use_constant = true;
      } else {
        if (value != "EXP"){
          std::stringstream msg;
          msg << "Error: Must specify either a constant or a second independent " <<
          "variable for the algrebra utility for user defined operation labeled: "<< label << std::endl;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__ );
        }
      }

      if ( value == "ADD" ) {
        new_op.expression_type = ADD;
      } else if ( value == "SUBTRACT" ){
        new_op.expression_type = SUBTRACT;
      } else if ( value == "MULTIPLY" ){
        new_op.expression_type = MULTIPLY;
      } else if ( value == "DIVIDE" ){
        new_op.expression_type = DIVIDE;
      } else if ( value == "DIVIDE_CONST_VARIABLE"){
        new_op.expression_type = DIVIDE_CONST_VARIABLE;
      } else if ( value == "DIVIDE_VARIABLE_CONST"){
        new_op.expression_type = DIVIDE_VARIABLE_CONST;
      } else if ( value =="POW" ){
        new_op.expression_type = POW;
      } else if ( value == "EXP" ){
        new_op.expression_type = EXP;
      } else {
        throw InvalidValue("Error: expression type not recognized",__FILE__,__LINE__);
      }

      //stuff into a map:
      all_operations[label] = new_op;

    }

    ProblemSpecP db_order = db->findBlock("exe_order");
    if ( db_order == 0 ){
      throw ProblemSetupException("Error: must specify an order of operations.",__FILE__,__LINE__);
    }
    for ( ProblemSpecP db_oneop = db_order->findBlock("op"); db_oneop !=  0;
          db_oneop = db_oneop->findNextBlock("op")){

      std::string label;
      db_oneop->getAttribute("label", label);
      order.push_back(label);

    }
  }

  // Local variable creation -----------------------------------------------------------------------
  template <typename T>
  void TaskAlgebra<T>::create_local_labels(){

    for ( auto iter = all_operations.begin(); iter != all_operations.end(); iter++ ){
      if ( iter->second.create_new_variable ){

        register_new_variable<T>(iter->second.dep);

      }
    }
  }

  // Initialize ------------------------------------------------------------------------------------
  template <typename T>
  void TaskAlgebra<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    for ( auto iter = all_operations.begin(); iter != all_operations.end(); iter++ ){
      if ( iter->second.create_new_variable ){

        register_variable( iter->second.dep, ArchesFieldContainer::COMPUTES, variable_registry );

      }
    }
  }

  template <typename T>
  void TaskAlgebra<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( typename OPMAP::iterator iter = all_operations.begin(); iter != all_operations.end(); iter++ ){
      if ( iter->second.create_new_variable ){

        T& dep = *(tsk_info->get_uintah_field<T>(iter->second.dep));
        Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
        Uintah::parallel_for( range, [&](int i, int j, int k){

          dep(i,j,k) = 0.;

        });
      }
    }
  }

  // Timestep initialize ---------------------------------------------------------------------------
  template <typename T>
  void TaskAlgebra<T>::register_timestep_init(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){

    for ( typename OPMAP::iterator iter = all_operations.begin();
          iter != all_operations.end(); iter++ ){

      if ( iter->second.create_new_variable ){

        register_variable( iter->second.dep, ArchesFieldContainer::COMPUTES, variable_registry );

      }
    }
  }

  template <typename T>
  void TaskAlgebra<T>::timestep_init(
    const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( typename OPMAP::iterator iter = all_operations.begin(); iter != all_operations.end(); iter++ ){
      if ( iter->second.create_new_variable ){

        T& dep = *(tsk_info->get_uintah_field<T>(iter->second.dep));
        Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
        Uintah::parallel_for( range, [&](int i, int j, int k){

          dep(i,j,k) = 0.;

        });
      }
    }
  }

  // Timestep work ---------------------------------------------------------------------------------
  template <typename T>
  void TaskAlgebra<T>::register_timestep_eval(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep ){

    std::vector<std::string> new_variables;
    std::vector<std::string> mod_variables;
    std::vector<std::string> req_variables;

    for (STRVEC::iterator iter = order.begin(); iter != order.end(); iter++){

      typename OPMAP::iterator op_iter = all_operations.find(*iter);
      if ( op_iter == all_operations.end() ){
        throw InvalidValue("Error: Operator not found: "+*iter, __FILE__,__LINE__ );
      }

      if ( op_iter->second.create_new_variable ){
        if ( !use_variable(op_iter->second.dep, new_variables)){
          register_variable( op_iter->second.dep, ArchesFieldContainer::MODIFIES, variable_registry );
          new_variables.push_back(op_iter->second.dep);
        }
      } else {
        if ( !use_variable(op_iter->second.dep, mod_variables)){
          register_variable( op_iter->second.dep, ArchesFieldContainer::MODIFIES, variable_registry );
          mod_variables.push_back(op_iter->second.dep);
        }
      }

      //require from newdw on everything else?
      if ( !use_variable(op_iter->second.ind1, req_variables) ){
        register_variable( op_iter->second.ind1, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
        req_variables.push_back(op_iter->second.ind1);
      }
      if ( !op_iter->second.use_constant ){
        if ( !use_variable(op_iter->second.ind2, req_variables) ){
          register_variable( op_iter->second.ind2, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry );
          req_variables.push_back(op_iter->second.ind2);
        }
      }
    }
  }

  template <typename T>
  void TaskAlgebra<T>::eval(
    const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for (STRVEC::iterator iter = order.begin(); iter != order.end(); iter++){

      typename OPMAP::iterator op_iter = all_operations.find(*iter);

      T& dep = *(tsk_info->get_uintah_field<T>(op_iter->second.dep));
      CT& ind1 = *(tsk_info->get_const_uintah_field<CT>(op_iter->second.ind1));
      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

      if ( op_iter->second.use_constant ){

        switch ( op_iter->second.expression_type ){
          case ADD:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = op_iter->second.constant + ind1(i,j,k);
            });
            break;
          case SUBTRACT:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = ind1(i,j,k) - op_iter->second.constant;
            });
            break;
          case MULTIPLY:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = ind1(i,j,k) * op_iter->second.constant;
            });
            break;
          case DIVIDE_VARIABLE_CONST:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = ind1(i,j,k) / op_iter->second.constant;
            });
            break;
          case DIVIDE_CONST_VARIABLE:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = (ind1(i,j,k) == 0) ? 0.0 : op_iter->second.constant / ind1(i,j,k);
            });
            break;
          case POW:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = std::pow(ind1(i,j,k),op_iter->second.constant);
            });
            break;
          case EXP:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = std::exp(ind1(i,j,k));
            });
            break;
          default:
            throw InvalidValue("Error: TaskAlgebra not supported.",__FILE__,__LINE__);
        }

      } else {

        CT& ind2 = *(tsk_info->get_const_uintah_field<CT>(op_iter->second.ind2));

        switch ( op_iter->second.expression_type ){
          case ADD:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = ind1(i,j,k) + ind2(i,j,k);
            });
            break;
          case SUBTRACT:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = ind1(i,j,k) - ind2(i,j,k);
            });
            break;
          case MULTIPLY:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = ind1(i,j,k) * ind2(i,j,k);
            });
            break;
          case DIVIDE:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = ind2(i,j,k) == 0 ? 0.0 : ind1(i,j,k) / ind2(i,j,k);
            });
            break;
          case POW:
            Uintah::parallel_for(range, [&](int i, int j, int k){
              dep(i,j,k) = std::pow(ind1(i,j,k), ind2(i,j,k));
            });
            break;
          default:
            throw InvalidValue("Error: TaskAlgebra not supported.",__FILE__,__LINE__);
        }
      }
    }
  }
}
#endif
