#ifndef Uintah_Component_Arches_Inject_h
#define Uintah_Component_Arches_Inject_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/Box.h>

/** 
 *  @class  Inject
 *  @author Jeremy Thornock
 *  @date   Aug, 2011
 *
 *  @brief A source term for "injecting" sources into transported variables in regions 
 *         of the flow specified by geom_objects
 *
 * @details 
   Values of constant source are added to the transport equation within prescribed geometric locations. The geometric 
   locations are set using the \code <geom_object> \endcode node.  Note that you can have several injectors for one defined constant source. 
   This code is templated to allow for source injection for all equation types.  

   The input file should look like this: 

   \code 
     <SourceTerms>
       <src label="user-defined-label" type="injector">
         <injector> 
           <geom_object> ... </geom_object>
         </injector>
         <constant> DOUBLE </constant>
       </src>
     </SourceTerms>
   \endcode

   The units of this source term is: [units of phi]/time/volume, where [units of phi] are the units of the transported variable. 
   Supported injector types are: 

   \code
    type="cc_inject_src"  -> for CCVariable
    type="fx_inject_src"  -> for SFCXVariable
    type="fy_inject_src"  -> for SFCZVariable
    type="fz_inject_src"  -> for SFCZVariable
   \endcode

 */

namespace Uintah { 

template < typename sT>
class Inject: public SourceTermBase {

public: 

  Inject<sT>( std::string srcName, SimulationStateP& shared_state, 
                       vector<std::string> reqLabelNames, std::string type );

  ~Inject<sT>();
  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);
  /** @brief Schedule the calculation of the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );
  /** @brief Actually compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );

  /** @brief Schedule initialization */ 
  void sched_initialize( const LevelP& level, SchedulerP& sched );
  void initialize( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw );

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, vector<std::string> required_label_names, SimulationStateP& shared_state ) 
        : _name(name), _shared_state(shared_state), _required_label_names(required_label_names){
          _type = "inject_src"; 
        };
      ~Builder(){}; 

      Inject<sT>* build()
      { return scinew Inject<sT>( _name, _shared_state, _required_label_names, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      SimulationStateP& _shared_state; 
      vector<std::string> _required_label_names; 

  }; // Builder
private:

  double d_constant; 
  std::vector<GeometryPieceP> _geomPieces; 

}; // end Inject

  // ===================================>>> Functions <<<========================================

  template <typename sT>
  Inject<sT>::Inject( std::string src_name, SimulationStateP& shared_state,
                              vector<std::string> req_label_names, std::string type ) 
  : SourceTermBase(src_name, shared_state, req_label_names, type)
  {
    _label_sched_init = false; 
    _src_label = VarLabel::create( src_name, sT::getTypeDescription() ); 

    if ( typeid(sT) == typeid(SFCXVariable<double>) )
      _source_grid_type = FX_SRC; 
    else if ( typeid(sT) == typeid(SFCYVariable<double>) )
      _source_grid_type = FY_SRC; 
    else if ( typeid(sT) == typeid(SFCZVariable<double>) )
      _source_grid_type = FZ_SRC; 
    else if ( typeid(sT) == typeid(CCVariable<double> ) ) {
      _source_grid_type = CC_SRC; 
    } else {
      throw InvalidValue( "Error: Attempting to instantiate source (Inject) with unrecognized type.", __FILE__, __LINE__); 
    }
  }
  
  template <typename sT>
  Inject<sT>::~Inject()
  {}
  
  //---------------------------------------------------------------------------
  // Method: Problem Setup
  //---------------------------------------------------------------------------
  template <typename sT>
  void Inject<sT>::problemSetup(const ProblemSpecP& inputdb)
  {
  
    ProblemSpecP db = inputdb; 
  
    for (ProblemSpecP inject_db = db->findBlock("injector"); inject_db != 0; inject_db = inject_db->findNextBlock("injector")){
  
      ProblemSpecP geomObj = inject_db->findBlock("geom_object");
      GeometryPieceFactory::create(geomObj, _geomPieces); 
  
    }

    db->getWithDefault("constant",d_constant, 0.); 
  
  }
  //---------------------------------------------------------------------------
  // Method: Schedule the calculation of the source term 
  //---------------------------------------------------------------------------
  template <typename sT>
  void Inject<sT>::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
  {
    std::string taskname = "Inject::eval";
    Task* tsk = scinew Task(taskname, this, &Inject::computeSource, timeSubStep);
  
    if (timeSubStep == 0 && !_label_sched_init) {
      // Every source term needs to set this flag after the varLabel is computed. 
      // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
      _label_sched_init = true;
  
      tsk->computes(_src_label);
    } else {
      tsk->modifies(_src_label); 
    }
  
    sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 
  
  }
  //---------------------------------------------------------------------------
  // Method: Actually compute the source term 
  //---------------------------------------------------------------------------
  template <typename sT>
  void Inject<sT>::computeSource( const ProcessorGroup* pc, 
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
      Box patchInteriorBox = patch->getBox(); 
  
      sT constSrc; 
      if ( new_dw->exists(_src_label, matlIndex, patch ) ){
        new_dw->getModifiable( constSrc, _src_label, matlIndex, patch ); 
        constSrc.initialize(0.0);
      } else {
        new_dw->allocateAndPut( constSrc, _src_label, matlIndex, patch );
        constSrc.initialize(0.0);
      } 
  
      // not sure which logic is best...
      // currently assuming that the # of geometry pieces is a small # so checking for patch/geometry piece 
      // intersection first rather than putting the cell iterator loop first. That way, we won't loop over every 
      // patch. 
  
      // loop over all geometry pieces
      CellIterator iter = patch->getCellIterator(); 
      if ( typeid(sT) == typeid(SFCXVariable<double>) )
        CellIterator iter = patch->getSFCXIterator(); 
      else if ( typeid(sT) == typeid(SFCYVariable<double>) )
        CellIterator iter = patch->getSFCYIterator(); 
      else if ( typeid(sT) == typeid(SFCZVariable<double>) )
        CellIterator iter = patch->getSFCZIterator(); 
      else if ( typeid(sT) != typeid(CCVariable<double> ) ) {
        // Bulletproofing
        proc0cout << " While attempting to compute: Inject.h " << endl;
        proc0cout << " Encountered a type mismatch error.  The current code cannot handle" << endl;
        proc0cout << " a type other than one of the following: " << endl;
        proc0cout << " 1) CCVariable<double> " << endl;
        proc0cout << " 2) SFCXVariable<double> " << endl;
        proc0cout << " 3) SFCYVariable<double> " << endl;
        proc0cout << " 4) SFCZVariable<double> " << endl;
        throw InvalidValue( "Please check the builder (probably in Arches.cc) and try again. ", __FILE__, __LINE__); 
      }

      for (unsigned int gp = 0; gp < _geomPieces.size(); gp++){
  
        GeometryPieceP piece = _geomPieces[gp];
        Box geomBox          = piece->getBoundingBox(); 
        Box b                = geomBox.intersect(patchInteriorBox); 
        
        // patch and geometry intersect
        if ( !( b.degenerate() ) ){
  
          // loop over all cells
          for (iter.begin(); !iter.done(); iter++){
            IntVector c = *iter; 
            
            Point p = patch->cellPosition( *iter );
            if ( piece->inside(p) ) {
  
              // add constant source if cell is inside geometry piece 
              constSrc[c] += d_constant; 
            }
          }
        }
      }
    }
  }
  
  //---------------------------------------------------------------------------
  // Method: Schedule dummy initialization
  //---------------------------------------------------------------------------
  template <typename sT>
  void Inject<sT>::sched_initialize( const LevelP& level, SchedulerP& sched )
  {
    string taskname = "Inject::initialize"; 
  
    Task* tsk = scinew Task(taskname, this, &Inject::initialize);
  
    tsk->computes(_src_label);
  
    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      tsk->computes(*iter); 
    }
  
    sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
  
  }
  template <typename sT>
  void Inject<sT>::initialize( const ProcessorGroup* pc, 
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
  
  
      sT src;
  
      new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 
  
      src.initialize(0.0); 
  
      // Note! Assuming that all dependent variables are the same class as the source. 
      for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
        sT tempVar; 
        new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
      }
    }
  }


} // end namespace Uintah
#endif
