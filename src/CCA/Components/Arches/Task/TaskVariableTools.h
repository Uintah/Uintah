#ifndef Uintah_Component_Arches_TaskVariableTools_h
#define Uintah_Component_Arches_TaskVariableTools_h

#include <CCA/Components/Arches/Task/FieldContainer.h>

//==================================================================================================

/**
* @class  Task Variable Tools
* @author Jeremy Thornock
* @date   2016
*
* @brief Tools associated with variable management within a Task.
*
**/

//==================================================================================================

namespace Uintah{

    /** @brief A container to hold a small amount of other information to
     *         pass into the task exe. **/
    struct SchedToTaskInfo{
      int time_substep;
      double dt;
    };

    /** @brief A class for managing the retrieval of uintah/so fields during task exe **/
    class ArchesTaskInfoManager{

      public:

        enum MAPCHECK {CHECK_FIELD,CONST_FIELD,NONCONST_FIELD};

        ArchesTaskInfoManager( std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                               const Patch* patch, SchedToTaskInfo& info ):
                        _var_reg(var_reg), _patch(patch), _tsk_info(info){

        };

        ~ArchesTaskInfoManager(){

        };

        /** @brief return the time substep **/
        inline int get_time_substep(){ return _tsk_info.time_substep; };

        /** @brief return the dt **/
        inline double get_dt(){ return _tsk_info.dt; };

        /** @brief return the variable registry **/
        inline std::vector<ArchesFieldContainer::VariableInformation>& get_variable_reg(){ return _var_reg; }

        /** @brief Set the references to the variable maps in the Field Collector for easier
         * management of the fields when trying to retrieve from the DW **/
        void set_field_container(ArchesFieldContainer* field_container){

          _field_container = field_container;

        }

        //====================================================================================
        // GRID VARIABLE ACCESS
        //====================================================================================
        /** @brief Return a CONST UINTAH field **/
        template <typename T>
        inline T* get_const_uintah_field( const std::string name ){
          return _field_container->get_const_field<T>(name);
        }

        /** @brief Return a CONST UINTAH field specifying the DW **/
        template <typename T>
        inline T* get_const_uintah_field( const std::string name,
          ArchesFieldContainer::WHICH_DW which_dw ){
          return _field_container->get_const_field<T>(name, which_dw);
        }

        /** @brief Return a UINTAH field **/
        template <typename T>
        inline T* get_uintah_field( const std::string name ){
          return _field_container->get_field<T>(name);
        }

        /** @brief Return a UINTAH particle field **/
        std::tuple<ParticleVariable<double>*, ParticleSubset*>
          get_uintah_particle_field( const std::string name ){
          return _field_container->get_uintah_particle_field( name );
        }

        /** @brief Return a const UINTAH particle field **/
        std::tuple<constParticleVariable<double>*, ParticleSubset*>
          get_const_uintah_particle_field( const std::string name ){
          return _field_container->get_const_uintah_particle_field( name );
        }

        /** @brief Get the current patch ID **/
        inline int get_patch_id(){ return _patch->getID(); }

      private:

        ArchesFieldContainer* _field_container;

        std::vector<ArchesFieldContainer::VariableInformation> _var_reg;
        const Patch* _patch;
        SchedToTaskInfo& _tsk_info;

    }; //End ArchesTaskInfoManager

    /** @brief Builds a struct for each variable containing all pertinent uintah
     * DW information **/
     static void register_variable_work( std::string name,
                                         ArchesFieldContainer::VAR_DEPEND dep,
                                         int nGhost,
                                         ArchesFieldContainer::WHICH_DW dw,
                                         ArchesFieldContainer::VariableRegistry& variable_registry,
                                         const int time_substep,
                                         const std::string task_name )
     {

      ArchesFieldContainer::VariableInformation info;

      info.name   = name;
      info.depend = dep;
      info.dw     = dw;
      info.nGhost = nGhost;
      info.local = false;

      info.is_constant = false;
      if ( dep == ArchesFieldContainer::REQUIRES ){
        info.is_constant = true;
      }

      switch (dw) {

      case ArchesFieldContainer::OLDDW:

        info.uintah_task_dw = Task::OldDW;
        break;

      case ArchesFieldContainer::NEWDW:

        info.uintah_task_dw = Task::NewDW;
        break;

      case ArchesFieldContainer::LATEST:

        if ( time_substep == 0 ){
          info.dw = ArchesFieldContainer::OLDDW;
          info.uintah_task_dw = Task::OldDW;
        } else {
          info.dw = ArchesFieldContainer::NEWDW;
          info.uintah_task_dw = Task::NewDW;
        }
        break;

      default:

        throw InvalidValue("Arches Task Error: Cannot determine the DW needed for variable: "+name, __FILE__, __LINE__);
        break;

      }

      //check for conflicts:
      if (dep == ArchesFieldContainer::COMPUTES && dw == ArchesFieldContainer::OLDDW) {
        throw InvalidValue("Arches Task Error: Cannot COMPUTE (ArchesFieldContainer::COMPUTES) a variable from OldDW for variable: "+name, __FILE__, __LINE__);
      }

      if ( (dep == ArchesFieldContainer::MODIFIES && dw == ArchesFieldContainer::OLDDW) ) {
        throw InvalidValue("Arches Task Error: Cannot MODIFY a variable from OldDW for variable: "+name, __FILE__, __LINE__);
      }

      if ( dep == ArchesFieldContainer::COMPUTES || dep == ArchesFieldContainer::MODIFIES ){

        if ( nGhost > 0 ) {

          std::cout << "Arches Task Warning: A variable COMPUTE/MODIFIES found that is requesting ghosts for: "+name+" Nghosts set to zero!" << std::endl;
          info.nGhost = 0;

        }
      }

      const VarLabel* the_label = nullptr;
      the_label = VarLabel::find( name );

      if ( the_label == nullptr ){
        throw InvalidValue("Error: The variable named: "+name+" does not exist for task: "+task_name,__FILE__,__LINE__);
      } else {
        info.label = the_label;
      }

      const Uintah::TypeDescription* type_desc = the_label->typeDescription();

      if ( dep == ArchesFieldContainer::REQUIRES ) {

        if ( nGhost == 0 ){
          info.ghost_type = Ghost::None;
        } else {
          if ( type_desc == CCVariable<int>::getTypeDescription() ) {
              info.ghost_type = Ghost::AroundCells;
          } else if ( type_desc == CCVariable<double>::getTypeDescription() ) {
              info.ghost_type = Ghost::AroundCells;
          } else if ( type_desc == CCVariable<Vector>::getTypeDescription() ) {
              info.ghost_type = Ghost::AroundCells;
          } else if ( type_desc == SFCXVariable<double>::getTypeDescription() ) {
              info.ghost_type = Ghost::AroundFaces;
          } else if ( type_desc == SFCYVariable<double>::getTypeDescription() ) {
              info.ghost_type = Ghost::AroundFaces;
          } else if ( type_desc == SFCZVariable<double>::getTypeDescription() ) {
              info.ghost_type = Ghost::AroundFaces;
          } else {
            throw InvalidValue("Error: No coverage yet for this type of variable.", __FILE__,__LINE__);
          }
        }
      }

      variable_registry.push_back( info );

    }

    /** @brief Inteface to register_variable_work -- this function is overloaded. **/
    static void register_variable( std::string name,
                                   ArchesFieldContainer::VAR_DEPEND dep,
                                   int nGhost,
                                   ArchesFieldContainer::WHICH_DW dw,
                                   std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                                   const int time_substep,
                                   std::string task_name="Var not registered with task name." ){
      register_variable_work( name, dep, nGhost, dw, var_reg, time_substep, task_name );
    }

    /** @brief Inteface to register_variable_work -- this function is overloaded. **/
    static void register_variable( std::string name,
                                   ArchesFieldContainer::VAR_DEPEND dep,
                                   int nGhost,
                                   ArchesFieldContainer::WHICH_DW dw,
                                   std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                                   std::string task_name="Var not registered with task name." ){
      register_variable_work( name, dep, nGhost, dw, var_reg, 0, task_name );
    }

    /** @brief Inteface to register_variable_work -- this function is overloaded.
     *         This version assumes NewDW and zero ghosts. **/
    static void register_variable( std::string name,
                                   ArchesFieldContainer::VAR_DEPEND dep,
                                   std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                                   std::string task_name="Var not registered with task name." ){

      ArchesFieldContainer::WHICH_DW dw = ArchesFieldContainer::NEWDW;
      int nGhost = 0;
      register_variable_work( name, dep, nGhost, dw, var_reg, 0, task_name );

    }

    /** @brief Inteface to register_variable_work -- this function is overloaded.
     *         This version assumes NewDW and zero ghosts and passes the timesubstep. **/
    static void register_variable( std::string name,
                                   ArchesFieldContainer::VAR_DEPEND dep,
                                   std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                                   const int timesubstep,
                                   std::string task_name="Task not registered with task name."){

      ArchesFieldContainer::WHICH_DW dw = ArchesFieldContainer::NEWDW;
      int nGhost = 0;
      register_variable_work( name, dep, nGhost, dw, var_reg, timesubstep, task_name );
    }

}

#endif
