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
        inline T* get_const_uintah_field( const std::string name , bool returnNullPointer=false ){
          T* emptyPointer=NULL;
          return returnNullPointer ? emptyPointer  :  _field_container->get_const_field<T>(name);
        }

        /** @brief Return a CONST UINTAH field **/
        template <typename T>
        inline T& get_const_uintah_field_add( const std::string name, bool returnNullPointer=false ){
          T* emptyPointer=NULL;
          return returnNullPointer ? *emptyPointer : *(_field_container->get_const_field<T>(name));
        }

        /** @brief Return a CONST UINTAH field specifying the DW **/
        template <typename T>
        inline T* get_const_uintah_field( const std::string name,
          ArchesFieldContainer::WHICH_DW which_dw ){
          return _field_container->get_const_field<T>(name, which_dw);
        }

        /** @brief Return a CONST UINTAH field specifying the DW **/
        template <typename T>
        inline T& get_const_uintah_field_add( const std::string name,
          ArchesFieldContainer::WHICH_DW which_dw ){
          return *(_field_container->get_const_field<T>(name, which_dw));
        }

        /** @brief Return a UINTAH field **/
        template <typename T>
        inline T* get_uintah_field( const std::string name , bool returnNullPointer=false ){
          T* emptyPointer=NULL;
          return returnNullPointer ? emptyPointer  :  _field_container->get_field<T>(name);
        }

        /** @brief Return a UINTAH field **/
        template <typename T>
        inline T& get_uintah_field_add( const std::string name, bool returnNullPointer=false ){
          T* emptyPointer=NULL;
          return returnNullPointer ? *emptyPointer  : *(_field_container->get_field<T>(name)) ;
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
     void register_variable_work( std::string name,
                                  ArchesFieldContainer::VAR_DEPEND dep,
                                  int nGhost,
                                  ArchesFieldContainer::WHICH_DW dw,
                                  ArchesFieldContainer::VariableRegistry& variable_registry,
                                  const int time_substep,
                                  const std::string task_name );

    /** @brief Inteface to register_variable_work -- this function is overloaded. **/
    void register_variable( std::string name,
                            ArchesFieldContainer::VAR_DEPEND dep,
                            int nGhost,
                            ArchesFieldContainer::WHICH_DW dw,
                            std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                            const int time_substep,
                            std::string task_name="(task name not not communicated to this variable registration)");

    /** @brief Inteface to register_variable_work -- this function is overloaded. **/
    void register_variable( std::string name,
                            ArchesFieldContainer::VAR_DEPEND dep,
                            int nGhost,
                            ArchesFieldContainer::WHICH_DW dw,
                            std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                            std::string task_name="(task name not not communicated to this variable registration)");

    /** @brief Inteface to register_variable_work -- this function is overloaded.
     *         This version assumes NewDW and zero ghosts. **/
    void register_variable( std::string name,
                            ArchesFieldContainer::VAR_DEPEND dep,
                            std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                            std::string task_name="(task name not not communicated to this variable registration)");

    /** @brief Inteface to register_variable_work -- this function is overloaded.
     *         This version assumes NewDW and zero ghosts and passes the timesubstep. **/
    void register_variable( std::string name,
                            ArchesFieldContainer::VAR_DEPEND dep,
                            std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                            const int timesubstep,
                            std::string task_name="(task name not not communicated to this variable registration)");

}

#endif
