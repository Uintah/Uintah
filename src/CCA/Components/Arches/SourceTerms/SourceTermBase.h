#ifndef Uintah_Component_Arches_SourceTermBase_h
#define Uintah_Component_Arches_SourceTermBase_h

#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <typeinfo>

//===============================================================

/**
* @class  SourceTermBase
* @author Jeremy Thornock
* @date   Nov, 5 2008
*
* @brief A base class for source terms for a transport
*        equation.
*
*/

namespace Uintah {

class BoundaryCondition;
class TableLookup;
class WBCHelper;

class SourceTermBase{

public:

  SourceTermBase( std::string srcName, MaterialManagerP& materialManager,
                  std::vector<std::string> reqLabelNames, std::string type );

  virtual ~SourceTermBase();

  /** @brief Indicates the var type of source this is **/
  enum MY_GRID_TYPE {CC_SRC, CCVECTOR_SRC, FX_SRC, FY_SRC, FZ_SRC};

  /** @brief Input file interface */
  virtual void problemSetup(const ProblemSpecP& db) = 0;

  /** @brief Returns a list of required variables from the DW for scheduling */
  //virtual void getDwVariableList() = 0;

  /** @brief Schedule the source for computation. */
  virtual void sched_computeSource(const LevelP& level, SchedulerP& sched, int timeSubStep ) = 0;

  /** @brief Actually compute the source. */
  virtual void computeSource( const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw,
                              int timeSubStep ) = 0;

  /** @brief  Initialize variables. */
  virtual void sched_initialize( const LevelP& level, SchedulerP& sched ) = 0;

  /** @brief  Initialize variables during a restart */
  virtual void sched_restartInitialize( const LevelP& level, SchedulerP& sched ){};

  // An optional call for the application to check their reduction vars.
  virtual void checkReductionVars( const ProcessorGroup * pg,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * matls,
                                         DataWarehouse  * old_dw,
                                         DataWarehouse  * new_dw ) {};

  /** @brief Work to be performed after properties are setup */
  virtual void extraSetup( GridP& grid, BoundaryCondition* bc, TableLookup* table_lookup ){ }

  /** @brief Returns a list of any extra local labels this source may compute **/
  inline const std::vector<const VarLabel*> getExtraLocalLabels(){
    return _extra_local_labels; }

  /** @brief Return the grid type of source (CC, FCX, etc... ) **/
  inline MY_GRID_TYPE getSourceGridType(){ return _source_grid_type; }

  /** @brief Return the type of source (constant, do_radation, etc... ) **/
  inline std::string getSourceType(){ return _type; }

  /** @brief Return an int indicating the stage this source should be executed **/
  int stage_compute() const { return _stage; }

  /** @brief Set the stage number **/
  void set_stage( const int stage );

  /** @brief Builder class containing instructions on how to build the property model **/
  class Builder {

    public:

      virtual ~Builder() {}

      virtual SourceTermBase* build() = 0;

    protected:

      std::string _name;
      std::string _type;
  };

  /** @brief In the case where a source base returns multiple sources, this returns that list.
              This is a limiting case with the current abstraction. **/
  std::vector<std::string>& get_all_src_names(){ return _mult_srcs; }

  void set_bcHelper( WBCHelper* helper ){
    m_bcHelper = helper; 
  }


protected:

  std::string _src_name;                                  ///< User assigned source name
  std::vector<std::string> _mult_srcs;                    ///< If a source produces multiple labels, this vector holds the name of each.
  std::string _init_type;                                 ///< Initialization type.
  std::string _type;                                      ///< Source type (eg, constant, westbrook dryer, .... )
  bool _compute_me;                                       ///< To indicate if calculating this source is needed or has already been computed.
  const VarLabel* _src_label;                             ///< Source varlabel
  const VarLabel* _simulationTimeLabel;
  int _stage;                                             ///< At which stage should this be computed: 0) before table lookup 1) after table lookup 2) after RK ave
  MaterialManagerP& _materialManager;                        ///< Local copy of materialManager
  std::vector<std::string> _required_labels;              ///< Vector of required labels
  std::vector<const VarLabel*> _extra_local_labels;       ///< Extra labels that might be useful for storage
  MY_GRID_TYPE _source_grid_type;                         ///< Source grid type
  TableLookup* _table_lookup;
  WBCHelper* m_bcHelper;


}; // end SourceTermBase
}  // end namespace Uintah

#endif
