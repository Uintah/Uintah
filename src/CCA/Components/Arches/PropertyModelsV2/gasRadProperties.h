#ifndef Uintah_Component_Arches_gasRadProperties_h
#define Uintah_Component_Arches_gasRadProperties_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
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


    void problemSetup( ProblemSpecP& db );

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info);

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info);

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();






    //Build instructions for this (CO) class.
      class Builder : public TaskInterface::TaskBuilder {

        public:

          Builder( std::string task_name, int matl_index)
            : _task_name(task_name), _matl_index(matl_index) {}
          ~Builder(){}

          gasRadProperties* build()
          { return scinew gasRadProperties( _task_name, _matl_index ); }

        private:

          std::string _task_name;
          int _matl_index;

      };

      //class Builder
        //: public TaskInterface::TaskBuilder { 

          //public: 

            //Builder( std::string name, SimulationStateP& shared_state,ArchesLabel * fieldLabels) : _name(name), _shared_state(shared_state),_fieldLabels(fieldLabels) {};
            //~Builder(){}; 

            //gasRadProperties* build()
            //{ return scinew gasRadProperties( _name, _shared_state, _fieldLabels ); };

          //private: 

            //std::string _name; 
            //SimulationStateP& _shared_state; 
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
