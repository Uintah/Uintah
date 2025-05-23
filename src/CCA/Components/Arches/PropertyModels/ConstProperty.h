#ifndef Uintah_Component_Arches_ConstProperty_h
#define Uintah_Component_Arches_ConstProperty_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>

/** 
* @class  ConstantProperty
* @author Jeremy Thornock
* @date   Aug. 2011
* 
* @brief A templated property model that is simply a constant value.  The templated
*        parameter is refering to the grid variable.  This derived class
*        should be a helpful template/example for adding additional property 
*        models. Please see PropertyModelBase.h for pure virtual function documentation.  
*        Any model specific functionality should be documented here. 
*        
* 
*/ 

namespace Uintah{ 

  template < typename pT, typename constpT>
  class ConstProperty : public PropertyModelBase {

    public: 

      ConstProperty<pT, constpT>( std::string prop_name, MaterialManagerP& materialManager );
      ~ConstProperty<pT, constpT>(); 

      void problemSetup( const ProblemSpecP& db ); 

      void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ); 
      void computeProp(const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw, 
                       int time_substep );

      void sched_initialize( const LevelP& level, SchedulerP& sched );
      void initialize( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw );

      class Builder
        : public PropertyModelBase::Builder { 

        public: 

          Builder( std::string name, MaterialManagerP& materialManager ) : _name(name), _materialManager(materialManager){};
          ~Builder(){}; 

          ConstProperty<pT, constpT>* build()
          { return scinew ConstProperty<pT, constpT>( _name, _materialManager ); };

        private: 

          std::string _name; 
          MaterialManagerP& _materialManager; 

      }; // class Builder 

    private: 

      double      _constant;                ///< Constant value from input file. 

  }; // class ConstProperty

  // ===================================>>> Functions <<<========================================
  
  template <typename pT, typename constpT>
  ConstProperty<pT, constpT>::ConstProperty( std::string prop_name, MaterialManagerP& materialManager ) : PropertyModelBase( prop_name, materialManager )
  {
    _prop_label = VarLabel::create( prop_name, pT::getTypeDescription() ); 

    // evaluate after table lookup: 
    _before_table_lookup = true; 
  }
  
  template <typename pT, typename constpT>
  ConstProperty<pT, constpT>::~ConstProperty( )
  {}
  
  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::problemSetup( const ProblemSpecP& inputdb )
  {
    ProblemSpecP db = inputdb; 

    db->getWithDefault("constant", _constant, 0.); 

    commonProblemSetup( inputdb ); 
  }
  
  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
  {

    std::string taskname = "ConstProperty::computeProp"; 
    Task* tsk = scinew Task( taskname, this, &ConstProperty::computeProp, time_substep ); 

    tsk->modifiesVar( _prop_label ); 

    sched->addTask( tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ) ); 

  }

  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::computeProp(const ProcessorGroup* pc, 
                                      const PatchSubset* patches, 
                                      const MaterialSubset* matls, 
                                      DataWarehouse* old_dw, 
                                      DataWarehouse* new_dw, 
                                      int time_substep )
  {
    //patch loop
    for (int p=0; p < patches->size(); p++){

      const Patch* patch = patches->get(p);
      int archIndex = 0;
      int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 

      pT prop; 
      new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 

      prop.initialize(_constant);

    }
  }
  
  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::sched_initialize( const LevelP& level, SchedulerP& sched )
  {
    std::string taskname = "ConstProperty::initialize"; 

    Task* tsk = scinew Task(taskname, this, &ConstProperty::initialize);
    tsk->computesVar(_prop_label); 

    sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
  }

  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::initialize( const ProcessorGroup* pc, 
                                      const PatchSubset* patches, 
                                      const MaterialSubset* matls, 
                                      DataWarehouse* old_dw, 
                                      DataWarehouse* new_dw )
  {
    //patch loop
    for (int p=0; p < patches->size(); p++){
  
      const Patch* patch = patches->get(p);
      int archIndex = 0;
      int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 

      pT prop; 

      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      prop.initialize(0.0); 

      PropertyModelBase::base_initialize( patch, prop ); 

    }
  }
}   // namespace Uintah

#endif
