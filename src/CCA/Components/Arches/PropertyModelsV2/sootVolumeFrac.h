#ifndef Uintah_Component_Arches_sootVolumeFrac_h
#define Uintah_Component_Arches_sootVolumeFrac_h
#include <Core/ProblemSpec/ProblemSpecP.h>
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
// * @brief Computes Soot volume fraction from Soot mass fraction. 
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

  class sootVolumeFrac : public TaskInterface{


 public:
    sootVolumeFrac( std::string prop_name, int matl_index  );
    //sootVolumeFrac( std::string task_name, int matl_index );
    ~sootVolumeFrac();

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
            : m_task_name(task_name), m_matl_index(matl_index) {}
          ~Builder(){}

          sootVolumeFrac* build()
          { return scinew sootVolumeFrac( m_task_name, m_matl_index ); }

        private:

          std::string m_task_name;
          int m_matl_index;

      };

      //class Builder
        //: public TaskInterface::TaskBuilder { 

          //public: 

            //Builder( std::string name, MaterialManagerP& materialManager,ArchesLabel * fieldLabels) : _name(name), _materialManager(materialManager),_fieldLabels(fieldLabels) {};
            //~Builder(){}; 

            //sootVolumeFrac* build()
            //{ return scinew sootVolumeFrac( _name, _materialManager, _fieldLabels ); };

          //private: 

            //std::string _name; 
            //MaterialManagerP& _materialManager; 
            //ArchesLabel* _fieldLabels;

        //}; // class Builder 




    private: 

     double _rho_soot;

     std::string _fvSoot;
     std::string _den_label_name;
     std::string _Ys_label_name; 
      
  }; // class sootVolumeFrac
}   // namespace Uintah

#endif
