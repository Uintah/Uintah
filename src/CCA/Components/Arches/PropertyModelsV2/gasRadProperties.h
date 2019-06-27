#ifndef Uintah_Component_Arches_gasRadProperties_h
#define Uintah_Component_Arches_gasRadProperties_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS
//
// /** 
// * @class  ADD
// * @author ADD
// * @date   ADD
// * 
// * @brief Computes ADD INFORMATION HERE
// *
// * ADD INPUT FILE INFORMATION HERE: 
// * The input file interface for this property should like this in your UPS file: 
// * \code 
// *   <PropertyModels>
// *     <.......>
// *   </PropertyModels>
// * \endcode 
// *  
// */ 

namespace Uintah{ 

  class RadPropertyCalculator; 

  class gasRadProperties : public TaskInterface{


 public:
    gasRadProperties( std::string prop_name, int matl_index  );
    //gasRadProperties( std::string task_name, int matl_index );
    ~gasRadProperties();

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();






    //Build instructions for this (CO) class.
      class Builder : public TaskInterface::TaskBuilder {

        public:

          Builder( std::string task_name, int matl_index)
            : m_task_name(task_name), m_matl_index(matl_index) {}
          ~Builder(){}

          gasRadProperties* build()
          { return scinew gasRadProperties( m_task_name, m_matl_index ); }

        private:

          std::string m_task_name;
          int m_matl_index;

      };

      //class Builder
        //: public TaskInterface::TaskBuilder { 

          //public: 

            //Builder( std::string name, MaterialManagerP& materialManager,ArchesLabel * fieldLabels) : _name(name), _materialManager(materialManager),_fieldLabels(fieldLabels) {};
            //~Builder(){}; 

            //gasRadProperties* build()
            //{ return scinew gasRadProperties( _name, _materialManager, _fieldLabels ); };

          //private: 

            //std::string _name; 
            //MaterialManagerP& _materialManager; 
            //ArchesLabel* _fieldLabels;

        //}; // class Builder 




    private: 
      
      double _absorption_modifier;
      RadPropertyCalculator::PropertyCalculatorBase* _calc; 
      const VarLabel* _temperature_label; 
      std::string _temperature_name;
      std::string _abskg_name;

  }; // class gasRadProperties
}   // namespace Uintah

#endif
