#ifndef Uintah_Component_Arches_TaskAlgebra_h
#define Uintah_Component_Arches_TaskAlgebra_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah {

  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename T>
  class TaskAlgebra : public TaskInterface {

public:

    enum EXPR {EQUALS, ADD, SUBTRACT, MULTIPLY, DIVIDE, DIVIDE_CONST_VARIABLE,
               DIVIDE_VARIABLE_CONST, POW, EXP};

    TaskAlgebra<T>( std::string task_name, int matl_index );
    ~TaskAlgebra<T>();

    void problemSetup( ProblemSpecP& db );

    //Build instructions for this (TaskAlgebra) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) :
        m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      TaskAlgebra* build()
      { return scinew TaskAlgebra<T>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

protected:

    void register_initialize(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const bool packed_tasks );

    void register_timestep_init(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const bool packed_tasks );

    void register_timestep_eval(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep,
      const bool packed_tasks );

    void register_compute_bcs(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep, const bool packed_tasks ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

private:

  typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

  struct Operation{

    std::string label;

    bool create_new_variable;
    bool create_temp_variable;
    bool ind1_is_temp;
    bool sum_into_dep;

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

    for ( ProblemSpecP op_db=db->findBlock("op"); op_db != nullptr; op_db=op_db->findNextBlock("op") ){

      Operation new_op;

      //get the name of this op
      std::string label;
      op_db->getAttribute("label",label);
      new_op.label = label;

      //does it create a new variable?
      new_op.create_new_variable = false;
      if ( op_db->findBlock("new_variable") ){
        new_op.create_new_variable = true;
      }

      new_op.sum_into_dep = false;
      if ( op_db->findBlock("sum_into_dep")){
        new_op.sum_into_dep = true;
      }

      //does it create a temp variable?
      new_op.create_temp_variable = false;
      if ( op_db->findBlock("dep_is_temp") ){
        new_op.create_temp_variable = true;
      }

      //does it use a temp variable?
      new_op.ind1_is_temp = false;
      if ( op_db->findBlock("ind1_is_temp") ){
        new_op.ind1_is_temp = true;
      }

      if ( new_op.ind1_is_temp && new_op.create_temp_variable ){
        throw ProblemSetupException("Error: One cannot ind1_is_temp and dep_is_temp for task_math op: "+label, __FILE__, __LINE__ );
      }

      if( new_op.create_temp_variable && new_op.create_new_variable ){
        throw ProblemSetupException("Error: Pick either TEMP or NEW variable creation (see arches_spec.xml) for task_math operation: "+label, __FILE__, __LINE__ );
      }

      //get variable names:
      op_db->require("dep", new_op.dep);
      op_db->require("ind1", new_op.ind1);

      //get the algebriac expression
      std::string value;
      op_db->getAttribute( "type", value );

      //optional ind2
      new_op.use_constant = false;
      if ( op_db->findBlock("ind2")){
        op_db->require("ind2", new_op.ind2);
      } else if ( op_db->findBlock("constant") ){
        op_db->require("constant", new_op.constant);
        new_op.use_constant = true;
      } else {
        if (value != "EXP" && value != "EQUALS"){
          std::stringstream msg;
          msg << "Error: Must specify either a constant or a second independent " <<
          "variable for the algrebra utility for user defined operation labeled: "<< label << std::endl;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__ );
        } else {
          new_op.use_constant = true; // not really, but used as a kludge
        }
      }

      if ( value == "EQUALS" ){
        new_op.expression_type = EQUALS;
      }
      else if ( value == "ADD" ) {
        new_op.expression_type = ADD;
      }
      else if ( value == "SUBTRACT" ){
        new_op.expression_type = SUBTRACT;
      }
      else if ( value == "MULTIPLY" ){
        new_op.expression_type = MULTIPLY;
      }
      else if ( value == "DIVIDE" ){
        new_op.expression_type = DIVIDE;
      }
      else if ( value == "DIVIDE_CONST_VARIABLE"){
        new_op.expression_type = DIVIDE_CONST_VARIABLE;
      }
      else if ( value == "DIVIDE_VARIABLE_CONST"){
        new_op.expression_type = DIVIDE_VARIABLE_CONST;
      }
      else if ( value =="POW" ){
        new_op.expression_type = POW;
      }
      else if ( value == "EXP" ){
        new_op.expression_type = EXP;
      }
      else {
        throw InvalidValue("Error: expression type not recognized",__FILE__,__LINE__);
      }

      //stuff into a map:
      all_operations[label] = new_op;

    }

    ProblemSpecP db_order = db->findBlock("exe_order");
    bool found_order = true;
    if ( db_order == nullptr ){
      found_order = false;
      //throw ProblemSetupException("Error: must specify an order of operations.",__FILE__,__LINE__);
    }

    if ( found_order ){
      //User specified a specific order of operations
      for ( ProblemSpecP db_oneop = db_order->findBlock("op");
            db_oneop !=  nullptr; db_oneop = db_oneop->findNextBlock("op")){

        std::string label;
        db_oneop->getAttribute("label", label);
        order.push_back(label);

      }
    } else {
      //Order of operations is arbitrary
      for ( auto i = all_operations.begin(); i != all_operations.end(); i++ ){
        order.push_back(i->second.label);
      }
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
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    for ( auto iter = all_operations.begin(); iter != all_operations.end(); iter++ ){
      if ( iter->second.create_new_variable ){

        register_variable( iter->second.dep, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

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
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    for ( typename OPMAP::iterator iter = all_operations.begin();
          iter != all_operations.end(); iter++ ){

      if ( iter->second.create_new_variable ){

        register_variable( iter->second.dep, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );

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
    const int time_substep, const bool packed_tasks ){

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
          register_variable( op_iter->second.dep, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
          new_variables.push_back(op_iter->second.dep);
        }
      } else {
        if ( !use_variable(op_iter->second.dep, mod_variables)){
          if ( !op_iter->second.create_temp_variable ){
            register_variable( op_iter->second.dep, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
            mod_variables.push_back(op_iter->second.dep);
          }
        }
      }

      //require from newdw on everything else?
      if ( !use_variable(op_iter->second.ind1, req_variables) ){
        if ( !op_iter->second.ind1_is_temp ){
          register_variable( op_iter->second.ind1, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
          req_variables.push_back(op_iter->second.ind1);
        }
      }
      if ( !op_iter->second.use_constant ){
        if ( !use_variable(op_iter->second.ind2, req_variables) ){
          register_variable( op_iter->second.ind2, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
          req_variables.push_back(op_iter->second.ind2);
        }
      }
    }
  }

  template <typename T>
  void TaskAlgebra<T>::eval(
    const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T temp_var;
    IntVector domlo = patch->getCellLowIndex();
    IntVector domhi = patch->getCellHighIndex();
    temp_var.allocate(domlo, domhi);
    temp_var.initialize(0.0);

    for (STRVEC::iterator iter = order.begin(); iter != order.end(); iter++){

      typename OPMAP::iterator op_iter = all_operations.find(*iter);

      T* dep_ptr;
      T* ind_ptr;

      if ( op_iter->second.ind1_is_temp ){
        ind_ptr = &temp_var;
      } else {
        ind_ptr = tsk_info->get_uintah_field<T>(op_iter->second.ind1);
      }

      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );

      if ( op_iter->second.create_temp_variable ) {
        dep_ptr = &temp_var;
      } else {
        dep_ptr = tsk_info->get_uintah_field<T>(op_iter->second.dep);
      }

      if ( op_iter->second.use_constant ){

        switch ( op_iter->second.expression_type ){
          case EQUALS:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k);
              });
            }
            break;
          case ADD:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + op_iter->second.constant + (*ind_ptr)(i,j,k);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = op_iter->second.constant + (*ind_ptr)(i,j,k);
              });
            }
            break;
          case SUBTRACT:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k) - op_iter->second.constant;
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) - op_iter->second.constant;
              });
            }
            break;
          case MULTIPLY:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k) * op_iter->second.constant;
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) * op_iter->second.constant;
              });
            }
            break;
          case DIVIDE_VARIABLE_CONST:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k) / op_iter->second.constant;
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) / op_iter->second.constant;
              });
            }
            break;
          case DIVIDE_CONST_VARIABLE:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + ((*ind_ptr)(i,j,k) == 0) ? 0.0
                  : op_iter->second.constant / (*ind_ptr)(i,j,k);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = ((*ind_ptr)(i,j,k) == 0) ? 0.0
                  : op_iter->second.constant / (*ind_ptr)(i,j,k);
              });
            }
            break;
          case POW:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + std::pow((*ind_ptr)(i,j,k),op_iter->second.constant);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = std::pow((*ind_ptr)(i,j,k),op_iter->second.constant);
              });
            }
            break;
          case EXP:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + std::exp((*ind_ptr)(i,j,k));
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = std::exp((*ind_ptr)(i,j,k));
              });
            }
            break;
          default:
            throw InvalidValue("Error: TaskAlgebra not supported.",__FILE__,__LINE__);
        }

      } else {

        T& ind2 = *(tsk_info->get_uintah_field<T>(op_iter->second.ind2));

        switch ( op_iter->second.expression_type ){
          case ADD:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k) + ind2(i,j,k);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) + ind2(i,j,k);
              });
            }
            break;
          case SUBTRACT:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k) - ind2(i,j,k);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) - ind2(i,j,k);
              });
            }
            break;
          case MULTIPLY:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k) * ind2(i,j,k);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) * ind2(i,j,k);
              });
            }
            break;
          case DIVIDE:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) == 0 ? 0.0 : (*dep_ptr)(i,j,k) + (*ind_ptr)(i,j,k) / ind2(i,j,k);
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*ind_ptr)(i,j,k) == 0 ? 0.0 : (*ind_ptr)(i,j,k) / ind2(i,j,k);
              });
            }
            break;
          case POW:
            if ( op_iter->second.sum_into_dep ){
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = (*dep_ptr)(i,j,k) + std::pow((*ind_ptr)(i,j,k), ind2(i,j,k));
              });
            } else {
              Uintah::parallel_for(range, [&](int i, int j, int k){
                (*dep_ptr)(i,j,k) = std::pow((*ind_ptr)(i,j,k), ind2(i,j,k));
              });
            }
            break;
          default:
            throw InvalidValue("Error: TaskAlgebra not supported.",__FILE__,__LINE__);
        }
      }
    }
  }
}
#endif
