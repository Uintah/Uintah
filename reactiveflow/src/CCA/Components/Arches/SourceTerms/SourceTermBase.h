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
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
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

class SourceTermBase{ 

public: 

  SourceTermBase( std::string srcName, SimulationStateP& sharedState, 
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

  /** @brief Work to be performed after properties are setup */ 
  virtual void extraSetup( GridP& grid ){ }

  /** @brief Returns the source label **/ 
  inline const VarLabel* getSrcLabel(){
    return _src_label; }

  /** @brief Returns a list of any extra local labels this source may compute **/ 
  inline const std::vector<const VarLabel*> getExtraLocalLabels(){
    return _extra_local_labels; }

  /** @brief Return the grid type of source (CC, FCX, etc... ) **/
  inline MY_GRID_TYPE getSourceGridType(){ return _source_grid_type; }

  /** @brief Return the type of source (constant, do_radation, etc... ) **/ 
  inline std::string getSourceType(){ return _type; }

  /** @brief Return the list of table lookup species needed for this source term **/ 
  inline ChemHelper::TableLookup* get_tablelookup_species(){ return _table_lookup_species; };  

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

protected:

  std::string _src_name;                                  ///< User assigned source name 
  std::string _init_type;                                 ///< Initialization type. 
  std::string _type;                                      ///< Source type (eg, constant, westbrook dryer, .... )
  bool _compute_me;                                       ///< To indicate if calculating this source is needed or has already been computed. 
  const VarLabel* _src_label;                             ///< Source varlabel
  int _stage;                                             ///< At which stage should this be computed: 0) before table lookup 1) after table lookup 2) after RK ave
  SimulationStateP& _shared_state;                        ///< Local copy of sharedState
  std::vector<std::string> _required_labels;              ///< Vector of required labels
  std::vector<const VarLabel*> _extra_local_labels;       ///< Extra labels that might be useful for storage
  ChemHelper::TableLookup* _table_lookup_species;         ///< List of table lookup species
  MY_GRID_TYPE _source_grid_type;                         ///< Source grid type
  std::vector<std::string> _wasatch_expr_names;           ///< List of wasatch exprs to be used as sources

}; // end SourceTermBase
}  // end namespace Uintah

#endif
