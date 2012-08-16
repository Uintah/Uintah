#ifndef Uintah_Component_Arches_CLASSNAME_h
#define Uintah_Component_Arches_CLASSNAME_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

// Instructions: 
//  1) Make sure you add doxygen comments!!
//  2) Do a find and replace on CLASSNAME to change the your class name 
//  3) This is a template!  Do a find on TEMP_PARAMS
//     to add your template parameters IN THE APPROPRIATE FORM to the function.
//     Note that you can't really do a find and replace here because TEMP_PARAMS will 
//     require slightly different information in different places. 
//  4) Make sure that your VarLabel for _src_label is declared with the 
//     correct template parameter along with any other templated VarLabels local
//     to this class 
//  5) Add implementaion details of your source term. 
//  6) Here is a brief checklist: 
//     a) Any extra grid variables for this source term need to be 
//        given VarLabels in the constructor
//     b) Any extra grid variable VarLabels need to be destroyed
//        in the local destructor
//     c) Add your input file details in problemSetup
//     d) Add actual calculation of source term in computeSource. 
//     e) Make sure that you dummyInit any new variables that require OldDW 
//        values.
//     f) Make sure that _before_table_lookup is properly set for this source term. 
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
* The input file interface for this source term should like this in your UPS file: 
* \code 
*   <Sources>
*     <src label="STRING" type="?????">
*         .....ADD DETAILS....
*     </src>
*   </Sources>
* \endcode 
*  
*/ 

namespace Uintah{

template < TEMP_PARAMS >
class CLASSNAME: public SourceTermBase {
public: 

  CLASSNAME<TEMP_PARAMS>( std::string srcName, SimulationStateP& shared_state, 
                          vector<std::string> reqLabelNames );
  ~CLASSNAME<TEMP_PARAMS>();

  void problemSetup(const ProblemSpecP& db);
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, vector<std::string> required_label_names, SimulationStateP& shared_state ) 
        : _name(name), _shared_state(shared_state), _required_label_names(required_label_names){};
      ~Builder(){}; 

      CLASSNAME<TEMP_PARAMS>* build()
      { return scinew CLASSNAME<TEMP_PARAMS>( _name, _shared_state, _required_label_names ); };

    private: 

      std::string _name; 
      SimulationStateP& _shared_state; 
      vector<std::string> _required_label_names; 

  }; // class Builder 

private:

}; // end CLASSNAME

  // ===================================>>> Functions <<<========================================

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
template<TEMP_PARAMS>
CLASSNAME<TEMP_PARAMS>::CLASSNAME( std::string src_name, SimulationStateP& shared_state,
                      vector<std::string> req_label_names ) 
: SourceTermBase( src_name, shared_state, req_label_names )
{
  _label_sched_init = false; 

  _src_label = VarLabel::create( src_name, sT::getTypeDescription() ); 

  // Add any other local variables here. 
  _extra_local_labels.resize(1); 
  _another_label = VarLabel::create( "string", TEMP_PARAMS::getTypeDescription() ); 
  _extra_local_labels[0] = _another_label;

  Declare the source type: 
  if ( typeid(sT) == typeid(SFCXVariable<double>) )
    _source_type = FX_SRC; 
  else if ( typeid(sT) == typeid(SFCYVariable<double>) )
    _source_type = FY_SRC; 
  else if ( typeid(sT) == typeid(SFCZVariable<double>) )
    _source_type = FZ_SRC; 
  else if ( typeid(sT) == typeid(CCVariable<double> ) ) {
    _source_type = CC_SRC; 
  else if ( typeid(sT) 
  } else {
    throw InvalidValue( "Error: Attempting to instantiate source (IntrusionInlet) with unrecognized type.", __FILE__, __LINE__); 
  }

  _source_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
template<TEMP_PARAMS>
CLASSNAME<TEMP_PARAMS>::~CLASSNAME()
{
  
  // source label is destroyed in the base class 

  for (vector<std::string>::iterator iter = _required_labels.begin(); 
       iter != _required_labels.end(); iter++) { 

    VarLabel::destroy( *iter ); 

  }

  // clean up any other stuff here
  
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
template <TEMP_PARAMS>
void CLASSNAME<TEMP_PARAMS>::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  // add input file interface here 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
template <TEMP_PARAMS>
void CLASSNAME<TEMP_PARAMS>::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CLASSNAME::eval";
  Task* tsk = scinew Task(taskname, this, &CLASSNAME::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);

  } else {

    tsk->modifies(_src_label); 

  }

  for (vector<std::string>::iterator iter = _required_labels.begin(); 
       iter != _required_labels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE SOURCe
    //tsk->requires( Task::OldDW, .... ); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
template <TEMP_PARAMS>
void CLASSNAME<TEMP_PARAMS>::computeSource( const ProcessorGroup* pc, 
                                            const PatchSubset* patches, 
                                            const MaterialSubset* matls, 
                                            DataWarehouse* old_dw, 
                                            DataWarehouse* new_dw, 
                                            int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    sT src; 
    if ( new_dw->exists( _src_label, matlIndex, patch ) ){
      new_dw->getModifiable( src, _src_label, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 
      src.initialize(0.0); 
    }

    // DEVELOPER'S NOTE:
    // One could in this case just initialize to the constant but we use a loop
    // to make this example a little more comprehensive. 
    // PLEASE NOTE the bulletproofing below.  Any new source term should have 
    // similar bulletproofing.   
    //

    CellIterator iter = patch->getCellIterator(); 
    if ( typeid(sT) == typeid(SFCXVariable<double>) )
      iter = patch->getSFCXIterator(); 
    else if ( typeid(sT) == typeid(SFCYVariable<double>) )
      iter = patch->getSFCYIterator(); 
    else if ( typeid(sT) == typeid(SFCZVariable<double>) )
      iter = patch->getSFCZIterator(); 
    else if ( typeid(sT) != typeid(CCVariable<double>) && 
              typeid(sT) != typeid(SFCXVariable<double>) &&
              typeid(sT) != typeid(SFCYVariable<double>) &&
              typeid(sT) != typeid(SFCZVariable<double>) ){
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

      src[*iter] = 0.0; //<---- do something here  

    }

  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
template <TEMP_PARAMS>
void CLASSNAME<TEMP_PARAMS>::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CLASSNAME::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &CLASSNAME::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++){

    tsk->computes(*iter); 

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
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

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
      tempVar.initialize(0.0); 
    }
  }
}


} // end namespace Uintah
#endif
