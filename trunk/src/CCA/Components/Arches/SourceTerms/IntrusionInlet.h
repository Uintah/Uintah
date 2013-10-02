#ifndef Uintah_Component_Arches_IntrusionInlet_h
#define Uintah_Component_Arches_IntrusionInlet_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/Box.h>

/** 
 * @class  Intrusion Inlet
 * @author Jeremy Thornock
 * @date   Aug 2011
 * 
 * @brief Adds a source term to cells adjacent to an intrusion to introduce sources
 *        of momentum, mass, and energy (like an inlet boundary condition). 
 *
 * The input file interface for this source term should like this in your UPS file: 
 *
 * \code 
 *   <Sources>
 *     <src label="STRING" type="TYPE">
 *       <intrusion>                              <!-- defines the intrusion (can have multiple)--> 
 *        <geom_object> ... </geom_object>         
 *       </intrusion>
 *       <normal>STRING</normal>                  <!-- defines the direction ( +/- X,Y,or Z) --> 
 *       <velocity>DOUBLE</velocity>              <!-- defines the inlet velocity --> 
 *       <density>DOUBLE</density>                <!-- defines the inlet density --> 
 *     </src>
 *   </Sources>
 * \endcode 
 *
 * Note that one may choose one of the following for source types depending on which 
 * equation this source is being used: 
 *   TYPE = cc_intrusion_inlet
 *   TYPE = fx_intrusion_inlet
 *   TYPE = fy_intrusion_inlet
 *   TYPE = fz_intrusion_inlet
 * 
 * @todo
 * Add a mass flowrate condition. 
 * Change interface when new table design is complete. 
 *  
 */ 

namespace Uintah{

  template < typename sT >
    class IntrusionInlet: public SourceTermBase {
      public: 

        enum DIRECTION { PLUS_X, MINUS_X, PLUS_Y, MINUS_Y, PLUS_Z, MINUS_Z }; 

        IntrusionInlet<sT>( std::string srcName, SimulationStateP& shared_state, 
            vector<std::string> reqLabelNames, std::string type );
        ~IntrusionInlet<sT>();

        void problemSetup(const ProblemSpecP& db);
        void sched_computeSource( const LevelP& level, SchedulerP& sched, 
            int timeSubStep );
        void computeSource( const ProcessorGroup* pc, 
            const PatchSubset* patches, 
            const MaterialSubset* matls, 
            DataWarehouse* old_dw, 
            DataWarehouse* new_dw, 
            int timeSubStep );
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
                  _type = "intrusion_inlet"; 
                };
              ~Builder(){}; 

              IntrusionInlet<sT>* build()
              { return scinew IntrusionInlet<sT>( _name, _shared_state, _required_label_names, _type ); };

            private: 

              std::string _name; 
              std::string _type; 
              SimulationStateP& _shared_state; 
              vector<std::string> _required_label_names; 

          }; // class Builder 

      private:

        std::vector<GeometryPieceP> _geomPieces;                     ///< vector of geometry pieces

        //double _mix_frac;                                            ///< inlet gas mixture fraction
        //double _heat_loss;                                           ///< inlet heat loss 
        double _velocity;                                            ///< inlet velocity
        double _density;                                             ///< inlet density
        double _phi;                                                 ///< value of phi (transported variable) at the inlet

        std::string _normal;                                         ///< normal direction of the inlet
        DIRECTION _dir; 


    }; // end IntrusionInlet

  // ===================================>>> Functions <<<========================================

  //---------------------------------------------------------------------------
  //Method: Constructor
  //---------------------------------------------------------------------------
  template<typename sT>
    IntrusionInlet<sT>::IntrusionInlet( std::string src_name, SimulationStateP& shared_state,
        vector<std::string> req_label_names, std::string type ) 
    : SourceTermBase( src_name, shared_state, req_label_names, type )
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
        throw InvalidValue( "Error: Attempting to instantiate source (IntrusionInlet) with unrecognized type.", __FILE__, __LINE__); 
      }
    }

  //---------------------------------------------------------------------------
  //Method: Destructor
  //---------------------------------------------------------------------------
  template<typename sT>
    IntrusionInlet<sT>::~IntrusionInlet()
    {}
  //---------------------------------------------------------------------------
  // Method: Problem Setup
  //---------------------------------------------------------------------------
  template <typename sT>
    void IntrusionInlet<sT>::problemSetup(const ProblemSpecP& inputdb)
    {

      ProblemSpecP db = inputdb; 

      // add input file interface here 
      int num_intrusions = 0; 
      for (ProblemSpecP intrusion_db = db->findBlock("intrusion"); 
          intrusion_db != 0; intrusion_db = intrusion_db->findNextBlock("intrusion")){

        ProblemSpecP geomObj = intrusion_db->findBlock("geom_object");
        GeometryPieceFactory::create(geomObj, _geomPieces); 

        ++num_intrusions; 

      }

      proc0cout << "Total number of intrusion inlets = " << num_intrusions << endl;

      //db->require( "mix_frac", _mix_frac ); 
      //db->require( "heat_loss", _heat_loss ); 
      db->require( "phi", _phi ); 
      db->require( "velocity", _velocity ); 
      db->require( "density",  _density ); 
      db->require( "normal", _normal); 

      if ( _normal == "+X" )
        _dir = PLUS_X;
      else if ( _normal == "-X" )
        _dir = MINUS_X; 
      else if ( _normal == "+Y" )
        _dir = PLUS_Y; 
      else if ( _normal == "-Y" )
        _dir = MINUS_Y; 
      else if ( _normal == "+Z" )
        _dir = PLUS_Z; 
      else if ( _normal == "-Z" )
        _dir = MINUS_Z; 

    }
  //---------------------------------------------------------------------------
  // Method: Schedule the calculation of the source term 
  //---------------------------------------------------------------------------
  template <typename sT>
    void IntrusionInlet<sT>::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
    {
      std::string taskname = "IntrusionInlet::eval";
      Task* tsk = scinew Task(taskname, this, &IntrusionInlet::computeSource, timeSubStep);

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
    void IntrusionInlet<sT>::computeSource( const ProcessorGroup* pc, 
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
        Vector Dx = patch->dCell(); 

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
          proc0cout << " While attempting to compute: IntrusionInlet.h " << endl;
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
              IntVector cn = *iter; // cell in the minus normal direction 
              double area = 0.0; 

              switch (_dir){

                case PLUS_X:
                  cn -= IntVector(1,0,0); 
                  area = Dx.y()*Dx.z(); 
                  break;
                case PLUS_Y:
                  cn -= IntVector(0,1,0); 
                  area = Dx.y()*Dx.z(); 
                  break;
                case PLUS_Z:
                  cn -= IntVector(0,0,1); 
                  area = Dx.x()*Dx.z(); 
                  break;
                case MINUS_X:
                  cn += IntVector(1,0,0); 
                  area = Dx.x()*Dx.z(); 
                  break;
                case MINUS_Y:
                  cn += IntVector(0,1,0); 
                  area = Dx.x()*Dx.y(); 
                  break;
                case MINUS_Z:
                  cn += IntVector(0,0,1); 
                  area = Dx.x()*Dx.y(); 
                  break;
                default:
                  throw InvalidValue("The specified normal direction was not one of: +X,-X,+Y,-Y,+Z,-Z",__FILE__,__LINE__); 

              }

              Point p = patch->cellPosition( c );
              Point pn = patch->cellPosition( cn ); 

              if ( !piece->inside(p) && piece->inside(pn) ) {

                src[c] += area * _density * _velocity * _phi; 

              }
            }
          }
        }
      }
    }

  //---------------------------------------------------------------------------
  // Method: Schedule initialization
  //---------------------------------------------------------------------------
  template <typename sT>
    void IntrusionInlet<sT>::sched_initialize( const LevelP& level, SchedulerP& sched )
    {
      string taskname = "IntrusionInlet::initialize"; 

      Task* tsk = scinew Task(taskname, this, &IntrusionInlet::initialize);

      tsk->computes(_src_label);

      for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
          iter != _extra_local_labels.end(); iter++){

        tsk->computes(*iter); 

      }

      sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

    }
  template <typename sT>
    void IntrusionInlet<sT>::initialize( const ProcessorGroup* pc, 
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
