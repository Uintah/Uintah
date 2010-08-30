#ifndef Uintah_Component_Arches_CLASSNAME_h
#define Uintah_Component_Arches_CLASSNAME_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>

// Instructions: 
//  1) Make sure you add doxygen comments!!
//  2) Do a find and replace on CLASSNAME to change the your class name 
//  3) This is a template!  Do a find on TEMP_PARAMS
//     to add your template parameters IN THE APPROPRIATE FORM to the function.
//     Note that you can't really do a find and replace here because TEMP_PARAMS will 
//     require slightly different information in different places.  See ConstProperty.h for 
//     an example. 
//  4) Make sure that your VarLabel for _prop_label is declared with the 
//     correct template parameter along with any other templated VarLabels local
//     to this class 
//  5) Add implementaion details of your property. 
//  6) Here is a brief checklist: 
//     a) Any extra grid variables for this property need to be 
//        given VarLabels in the constructor
//     b) Any extra grid variable VarLabels need to be destroyed
//        in the local destructor
//     c) Add your input file details in problemSetup
//     d) Add actual calculation of property in computeProp. 
//     e) Make sure that you dummyInit any new variables that require OldDW 
//        values.
//     f) Make sure that _before_table_lookup is properly set for this property. 
//   7) Please clean up unused code from this template in your final version
//   8) Please add comments to this list as you see fit to help the next person

/** 
* @class  ADD
* @author ADD
* @date   ADD
* 
* @brief Computes ADD INFORMATION HERE
*
* ADD INPUT FILE INFORMATION HERE: 
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <.......>
*   </PropertyModels>
* \endcode 
*  
*/ 

namespace Uintah{ 

  template < TEMP_PARAMS >
  class CLASSNAME : public PropertyModelBase {

    public: 

      CLASSNAME<TEMP_PARAMS>( std::string prop_name, SimulationStateP& shared_state );
      ~CLASSNAME<TEMP_PARAMS>(); 

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

          CLASSNAMETEMP_PARAMS* build()
          { return scinew CLASSNAMETEMP_PARAMS( _name, _shared_state ); };

        private: 

          std::string _name; 
          SimulationStateP& _shared_state; 

      }; // class Builder 

    private: 

  }; // class CLASSNAME

  // ===================================>>> Functions <<<========================================
  
  //---------------------------------------------------------------------------
  //Method: Constructor
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  CLASSNAME<TEMP_PARAMS>::CLASSNAME( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
  {

    _prop_label = VarLabel::create( prop_name, pT::getTypeDescription() ); 

    // additional local labels as needed by this class (delete this if it isn't used): 
    std::string name = "something"; 
    _something_label = VarLabel::create( name, TEMP_PARAMS::getTypeDescription() ); // Note: you need to add the label to the .h file
    _extra_local_labels.push_back( _something_label ); 

    // and so on ....

    // Evaluated before or after table lookup? 
    _before_table_lookup = false; 
  }
  
  //---------------------------------------------------------------------------
  //Method: Destructor
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  CLASSNAME<TEMP_PARAMS>::~CLASSNAME( )
  {
    // Destroying all local VarLabels stored in _extra_local_labels: 
    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

      VarLabel::destroy( *iter ); 

    }
    // Clean up anything else here ... 
  }
  
  //---------------------------------------------------------------------------
  //Method: Problem Setup
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  void CLASSNAME<TEMP_PARAMS>::problemSetup( const ProblemSpecP& inputdb )
  {
    ProblemSpecP db = inputdb; 
  }
  
  //---------------------------------------------------------------------------
  //Method: Schedule Compute Property
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  void CLASSNAME<TEMP_PARAMS>::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
  {
    std::string taskname = "CLASSNAME::computeProp"; 
    Task* tsk = scinew Task( taskname, this, &CLASSNAME::computeProp, time_substep ); 

    if ( !(_has_been_computed) ) {

      if ( time_substep == 0 ) {
        
        tsk->computes( _prop_label ); 

      } else {

        tsk->modifies( _prop_label ); 

      }

      sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
      
      _has_been_computed = true; 

    }
  }

  //---------------------------------------------------------------------------
  //Method: Actually Compute Property
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  void CLASSNAME<TEMP_PARAMS>::computeProp(const ProcessorGroup* pc, 
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
      else if ( typeid(pT) != typeid(CCVariable<double>) && 
                typeid(pT) != typeid(SFCXVariable<double>) &&
                typeid(pT) != typeid(SFCYVariable<double>) &&
                typeid(pT) != typeid(SFCZVariable<double>) ){
        // Bulletproofing
        proc0cout << " While attempting to compute: CLASSNAME.h " << endl;
        proc0cout << " Encountered a type mismatch error.  The current code cannot handle" << endl;
        proc0cout << " a type other than one of the following: " << endl;
        proc0cout << " 1) CCVariable<double> " << endl;
        proc0cout << " 2) SFCXVariable<double> " << endl;
        proc0cout << " 3) SFCYVariable<double> " << endl;
        proc0cout << " 4) SFCZVariable<double> " << endl;
        throw InvalidValue( "Please check the builder (probably in Arches.cc) and try again. ", __FILE__, __LINE__); 
      }

      for (iter.begin(); !iter.done(); iter++){

        prop[*iter] = 0.0; //<---- do something here  

      }
    }
  }
  
  //---------------------------------------------------------------------------
  //Method: Scheduler for Dummy Initialization
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  void CLASSNAME<TEMP_PARAMS>::sched_dummyInit( const LevelP& level, SchedulerP& sched )
  {

    std::string taskname = "CLASSNAME::dummyInit"; 

    Task* tsk = scinew Task(taskname, this, &CLASSNAME::dummyInit);
    tsk->computes(_prop_label); 
    tsk->requires( Task::OldDW, _prop_label, Ghost::None, 0 ); 

    sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

  }

  //---------------------------------------------------------------------------
  //Method: Actually do the Dummy Initialization
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  void CLASSNAME<TEMP_PARAMS>::dummyInit( const ProcessorGroup* pc, 
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

      //prop.initialize(0.0); <--- Careful, don't reinitialize if you don't want to 
      prop.copyData( old_prop );
  
    }
  }

  //---------------------------------------------------------------------------
  //Method: Scheduler for Initializing the Property
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  void CLASSNAME<TEMP_PARAMS>::sched_initialize( const LevelP& level, SchedulerP& sched )
  {
    std::string taskname = "CLASSNAME::initialize"; 

    Task* tsk = scinew Task(taskname, this, &CLASSNAME::initialize);
    tsk->computes(_prop_label); 

    sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
  }

  //---------------------------------------------------------------------------
  //Method: Actually Initialize the Property
  //---------------------------------------------------------------------------
  template <TEMP_PARAMS>
  void CLASSNAME<TEMP_PARAMS>::initialize( const ProcessorGroup* pc, 
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

      PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 

    }
  }
  

}   // namespace Uintah

#endif
