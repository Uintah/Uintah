#ifndef Uintah_Component_Arches_ConstProperty_h
#define Uintah_Component_Arches_ConstProperty_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

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

      ConstProperty<pT, constpT>( std::string prop_name, SimulationStateP& shared_state );
      ~ConstProperty<pT, constpT>(); 

      void problemSetup( const ProblemSpecP& db ); 

      void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ); 
      void computeProp(const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw, 
                       int time_substep );

      void sched_dummyInit( const LevelP& level, SchedulerP& sched );
      void dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw );

      void sched_initialize( const LevelP& level, SchedulerP& sched );
      void initialize( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw );

      class Builder
        : public PropertyModelBase::Builder { 

        public: 

          Builder( std::string name, SimulationStateP& shared_state ) : _name(name), _shared_state(shared_state){};
          ~Builder(){}; 

          ConstProperty<pT, constpT>* build()
          { return scinew ConstProperty<pT, constpT>( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

      double      _constant;                ///< Constant value from input file. 

  }; // class ConstProperty

  // ===================================>>> Functions <<<========================================
  
  template <typename pT, typename constpT>
  ConstProperty<pT, constpT>::ConstProperty( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
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
  }
  
  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
  {
    std::string taskname = "ConstProperty::computeProp"; 
    Task* tsk = scinew Task( taskname, this, &ConstProperty::computeProp, time_substep ); 

    if ( !(_has_been_computed) ) {

      if ( time_substep == 0 ) {
        
        tsk->computes( _prop_label ); 

      } else {

        tsk->modifies( _prop_label ); 

      }

      if ( !(_has_been_computed ) ) 
        sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
      
      _has_been_computed = true; 

    }
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
      int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

      pT prop; 
      if ( new_dw->exists( _prop_label, matlIndex, patch ) ){
        new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 
      } else {
        new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
        prop.initialize(0.0); 
      }

      // DEVELOPER'S NOTE:
      // One could in this case just initialize to the constant but we use a loop
      // to make this example a little more comprehensive. 
      // PLEASE NOTE the bulletproofing below.  Any new property model should have 
      // similar bulletproofing.   
      //

      CellIterator iter = patch->getCellIterator(); 
      if ( typeid(pT) == typeid(SFCXVariable<double>) )
        iter = patch->getSFCXIterator(); 
      else if ( typeid(pT) == typeid(SFCYVariable<double>) )
        iter = patch->getSFCYIterator(); 
      else if ( typeid(pT) == typeid(SFCZVariable<double>) )
        iter = patch->getSFCZIterator(); 
      else if ( typeid(pT) == typeid(CCVariable<double>) )
        iter = patch->getCellIterator(); 
      else {
        // Bulletproofing
        proc0cout << " While attempting to compute: ConstProperty.h " << endl;
        proc0cout << " Encountered a type mismatch error.  The current code cannot handle" << endl;
        proc0cout << " a type other than one of the following: " << endl;
        proc0cout << " 1) CCVariable<double> " << endl;
        proc0cout << " 2) SFCXVariable<double> " << endl;
        proc0cout << " 3) SFCYVariable<double> " << endl;
        proc0cout << " 4) SFCZVariable<double> " << endl;
        throw InvalidValue( "Please check the builder (probably in Arches.cc) and try again. ", __FILE__, __LINE__); 
      }

      for (iter.begin(); !iter.done(); iter++){

        prop[*iter] = _constant; 

      }
    }
  }
  
  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::sched_dummyInit( const LevelP& level, SchedulerP& sched )
  {

    std::string taskname = "ConstProperty::dummyInit"; 

    Task* tsk = scinew Task(taskname, this, &ConstProperty::dummyInit);
    tsk->computes(_prop_label); 
    tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

    sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

  }

  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::dummyInit( const ProcessorGroup* pc, 
                                              const PatchSubset* patches, 
                                              const MaterialSubset* matls, 
                                              DataWarehouse* old_dw, 
                                              DataWarehouse* new_dw )
  {
    //patch loop
    for (int p=0; p < patches->size(); p++){
  
      const Patch* patch = patches->get(p);
      int archIndex = 0;
      int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

      pT prop; 
      constpT old_prop; 

      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      old_dw->get( old_prop, _prop_label, matlIndex, patch, Ghost::None, 0); 

      prop.initialize(0.0); 
      prop.copyData( old_prop );
  
    }
  }

  template <typename pT, typename constpT>
  void ConstProperty<pT, constpT>::sched_initialize( const LevelP& level, SchedulerP& sched )
  {
    std::string taskname = "ConstProperty::initialize"; 

    Task* tsk = scinew Task(taskname, this, &ConstProperty::initialize);
    tsk->computes(_prop_label); 

    sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
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
      int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

      pT prop; 

      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      prop.initialize(0.0); 

      PropertyModelBase::base_initialize( patch, prop ); 

    }
  }
}   // namespace Uintah

#endif
