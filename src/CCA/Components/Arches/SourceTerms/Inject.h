#ifndef Uintah_Component_Arches_Inject_h
#define Uintah_Component_Arches_Inject_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>

/**
 *  @class  Inject
 *  @author Jeremy Thornock
 *  @date   April 20
 *
 *  @brief A source term for "injecting" sources into transported variables in regions
 *         of the flow specified by Interior boundary conditions
 *
 * @details
   Values of constant source are added to the transport equation given an interior face iterator.

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

  Inject<sT>( std::string srcName, MaterialManagerP& materialManager,
                       std::vector<std::string> reqLabelNames, std::string type );

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

      Builder( std::string name, std::vector<std::string> required_label_names, MaterialManagerP& materialManager )
        : _name(name), _materialManager(materialManager), _required_label_names(required_label_names){
          _type = "inject_src";
        };
      ~Builder(){};

      Inject<sT>* build()
      { return scinew Inject<sT>( _name, _materialManager, _required_label_names, _type ); };

    private:

      std::string _name;
      std::string _type;
      MaterialManagerP& _materialManager;
      std::vector<std::string> _required_label_names;


  }; // Builder
private:

  typedef std::map<IntVector, double> CellToValueMap;

  CellToValueMap _storage;
  IntVector m_rel_ijk;
  Point m_rel_xyz;

  CellToValueMap readInputFile( std::string file_name );

}; // end Inject

  // ===================================>>> Functions <<<========================================

  template <typename sT>
  Inject<sT>::Inject( std::string src_name, MaterialManagerP& materialManager,
                              std::vector<std::string> req_label_names, std::string type )
  : SourceTermBase(src_name, materialManager, req_label_names, type)
  {
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

    if ( db->findBlock("inputfile") ){

      std::string file_name;
      db->require("inputfile", file_name);

      gzFile file = gzopen( file_name.c_str(), "r" );

      if ( file == nullptr ) {
        proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << std::endl;
        throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
      }

      _storage = Inject::readInputFile( file_name );

      db->require("relative_xyz", m_rel_xyz);

    }

  }
  //---------------------------------------------------------------------------
  // Method: Schedule the calculation of the source term
  //---------------------------------------------------------------------------
  template <typename sT>
  void Inject<sT>::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
  {
    std::string taskname = "Inject::eval";
    Task* tsk = scinew Task(taskname, this, &Inject::computeSource, timeSubStep);

    if (timeSubStep == 0) {
      tsk->computes(_src_label);
    } else {
      tsk->modifies(_src_label);
    }

    sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

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
      int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

      IntVector ijk = patch->getLevel()->getCellIndex(m_rel_xyz);

      sT src;
      if ( new_dw->exists(_src_label, matlIndex, patch ) ){
        new_dw->getModifiable( src, _src_label, matlIndex, patch );
        src.initialize(0.0);
      } else {
        new_dw->allocateAndPut( src, _src_label, matlIndex, patch );
        src.initialize(0.0);
      }

      //==============================================
      const BndMapT& bc_info = m_bcHelper->get_boundary_information();
      for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

        std::string facename = i_bc->second.name;
        Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

        const BndCondSpec* spec = i_bc->second.find(_src_name);
        Uintah::Patch::FaceType face = i_bc->second.face;

        BndEdgeType edge_type = i_bc->second.edge_type;

        if ( spec != nullptr ){

          if (edge_type == BndEdgeType::INTERIOR ){

            if ( spec->bcType == DIRICHLET ){

              int i0; int i1; int zeroi;
              if ( face == Patch::xminus || face == Patch::xplus ){
                i0 = 1;
                i1 = 2;
                zeroi = 0;
              } else if ( face == Patch::yminus || face == Patch::yplus ){
                i0 = 2;
                i1 = 0;
                zeroi = 1;
              } else {
                i0 = 0;
                i1 = 1;
                zeroi = 2;
              }

              Vector DX = patch->dCell();

              double area = DX[i0]*DX[i1];
              double vol = DX[i0]*DX[i1]*DX[zeroi];

              const double value = spec->value;

              const int mysize = cell_iter.size();
              ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> cpuExeobj; //using temporary exeObj with CPU space. Remove while porting the task
              auto this_iter = cell_iter.get_ref_to_iterator(cpuExeobj);

              for ( int i = 0; i < mysize; i++ ){
                IntVector c = IntVector(this_iter[i][0], this_iter[i][1], this_iter[i][2]);
                src[IntVector(this_iter[i][0], this_iter[i][1], this_iter[i][2])] = value*area/vol;
              }

            } else if ( spec->bcType == CUSTOM ){

              int i0; int i1; int zeroi;
              if ( face == Patch::xminus || face == Patch::xplus ){
                i0 = 1;
                i1 = 2;
                zeroi = 0;
              } else if ( face == Patch::yminus || face == Patch::yplus ){
                i0 = 2;
                i1 = 0;
                zeroi = 1;
              } else {
                i0 = 0;
                i1 = 1;
                zeroi = 2;
              }

              Vector DX = patch->dCell();

              double area = DX[i0]*DX[i1];
              double vol = DX[i0]*DX[i1]*DX[zeroi];

              const int mysize = cell_iter.size();
              ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> cpuExeobj; //using temporary exeObj with CPU space. Remove while porting the task
              auto this_iter = cell_iter.get_ref_to_iterator(cpuExeobj);

              for ( int i = 0; i < mysize; i++ ){
                IntVector lookup_c = IntVector(this_iter[i][0], this_iter[i][1], this_iter[i][2]) - ijk;
                lookup_c[zeroi] = 0;

                auto ptr = _storage.find(IntVector(lookup_c));
                if ( ptr != _storage.end() ){
                  double value = ptr->second;
                  src[IntVector(this_iter[i][0], this_iter[i][1], this_iter[i][2])] = value*area/vol;
                }

              }
            } else {
              throw InvalidValue("Error: Boundary spec not recognized for source: "+_src_name, __FILE__, __LINE__);
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
  void Inject<sT>::sched_initialize( const LevelP& level, SchedulerP& sched )
  {
    std::string taskname = "Inject::initialize";

    Task* tsk = scinew Task(taskname, this, &Inject::initialize);

    tsk->computes(_src_label);

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      tsk->computes(*iter);
    }

    sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

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
      int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

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

  template <typename sT>
  std::map<IntVector, double>
  Inject<sT>::readInputFile( std::string file_name )
  {

    gzFile file = gzopen( file_name.c_str(), "r" );
    if ( file == nullptr ) {
      proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << std::endl;
      throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
    }

    std::string variable = getString( file );
    //double space1 = getDouble( file );
    //double space2 = getDouble( file );
    int num_points = getInt( file );
    std::map<IntVector, double> result;

    for ( int i = 0; i < num_points; i++ ) {
      int I = getInt( file );
      int J = getInt( file );
      int K = getInt( file );
      double v = getDouble( file );

      IntVector C(I,J,K);

      result.insert( std::make_pair( C, v ));

    }

    gzclose( file );
    return result;
  }


} // end namespace Uintah
#endif
