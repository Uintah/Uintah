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
 *  @date   Aug, 2010
 *
 *  @brief A source term for "injecting" sources into transported variables in regions 
 *         of the flow specified by geom_objects
 *
 * @details 
   Values of constant source are added to the transport equation within prescribed geometric locations. The geometric 
   locations are set using the <geom_object> node.  Note that you can have several injectors for one defined constant source. 
   This code is templated to allow for source injection for all equation types.  

   The input file should look like this: 

   \code 
     <SourceTerms>
       <src label="user-defined-label" type="injector">
         <injector> 
           <geom_object> ... </geom_object>
         </injector>
         <constant> 1.0 </constant>
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
                       vector<std::string> reqLabelNames );

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

  /** @brief Schedule a dummy initialization */ 
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
        : _name(name), d_sharedState(shared_state), _required_label_names(required_label_names){};
      ~Builder(){}; 

      Inject<sT>* build()
      { return scinew Inject<sT>( _name, d_sharedState, _required_label_names ); };

    private: 

      std::string _name; 
      SimulationStateP& d_sharedState; 
      vector<std::string> _required_label_names; 

  }; // Builder
private:

  double d_constant; 
  std::vector<GeometryPieceP> d_geomPieces; 

}; // end Inject

  // ===================================>>> Functions <<<========================================

  template <typename sT>
  Inject<sT>::Inject( std::string srcName, SimulationStateP& shared_state,
                              vector<std::string> req_label_names ) 
  : SourceTermBase(srcName, shared_state, req_label_names)
  {
    d_srcLabel = VarLabel::create( srcName, sT::getTypeDescription() ); 
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
      GeometryPieceFactory::create(geomObj, d_geomPieces); 
  
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
  
    if (timeSubStep == 0 && !d_labelSchedInit) {
      // Every source term needs to set this flag after the varLabel is computed. 
      // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
      d_labelSchedInit = true;
  
      tsk->computes(d_srcLabel);
    } else {
      tsk->modifies(d_srcLabel); 
    }
  
    sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
  
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
      int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
      Box patchInteriorBox = patch->getBox(); 
  
      sT constSrc; 
      if ( new_dw->exists(d_srcLabel, matlIndex, patch ) ){
        new_dw->getModifiable( constSrc, d_srcLabel, matlIndex, patch ); 
        constSrc.initialize(0.0);
      } else {
        new_dw->allocateAndPut( constSrc, d_srcLabel, matlIndex, patch );
        constSrc.initialize(0.0);
      } 
  
      // not sure which logic is best...
      // currently assuming that the # of geometry pieces is a small # so checking for patch/geometry piece 
      // intersection first rather than putting the cell iterator loop first. That way, we won't loop over every 
      // patch. 
  
      // loop over all geometry pieces
      CellIterator iter = patch->getCellIterator(); 
      if ( typeid(sT) == typeid(SFCXVariable<double>) )
        iter = patch->getSFCXIterator(); 
      else if ( typeid(sT) == typeid(SFCYVariable<double>) )
        iter = patch->getSFCYIterator(); 
      else if ( typeid(sT) == typeid(SFCZVariable<double>) )
        iter = patch->getSFCZIterator(); 
      else {
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

      for (int gp = 0; gp < d_geomPieces.size(); gp++){
  
        GeometryPieceP piece = d_geomPieces[gp];
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
  void Inject<sT>::sched_dummyInit( const LevelP& level, SchedulerP& sched )
  {
    string taskname = "Inject::dummyInit"; 
  
    Task* tsk = scinew Task(taskname, this, &Inject::dummyInit);
  
    tsk->computes(d_srcLabel);
  
    for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
      tsk->computes(*iter); 
    }
  
    sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
  
  }
  template <typename sT>
  void Inject<sT>::dummyInit( const ProcessorGroup* pc, 
                                       const PatchSubset* patches, 
                                       const MaterialSubset* matls, 
                                       DataWarehouse* old_dw, 
                                       DataWarehouse* new_dw )
  {
    //patch loop
    for (int p=0; p < patches->size(); p++){
  
      const Patch* patch = patches->get(p);
      int archIndex = 0;
      int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
  
  
      sT src;
  
      new_dw->allocateAndPut( src, d_srcLabel, matlIndex, patch ); 
  
      src.initialize(0.0); 
  
      // Note! Assuming that all dependent variables are the same class as the source. 
      for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
        sT tempVar; 
        new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
      }
    }
  }


} // end namespace Uintah
#endif
