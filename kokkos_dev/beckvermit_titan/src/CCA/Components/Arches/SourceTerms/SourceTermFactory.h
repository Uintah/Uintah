#ifndef UT_SourceTermFactory_h
#define UT_SourceTermFactory_h

#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h> 
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationStateP.h>
#include <map>
#include <vector>
#include <string>

/**
 *  @class  SourceTermFactory
 *  @author Jeremy Thornock
 *  @date   Aug 2011
 *  @brief  Factory for source term generation
 *
 *  Allows easy addition of source terms to a transport equation.
 *  Simply register the builder object for your SourceTerm with
 *  the factory and it will automatically be added to the requested
 *  TransportEquation.  Multiple source terms may be registered with a
 *  single transport equation.
 *
 *  Implemented as a singleton.
 */
 namespace Uintah {

   class ArchesLabel; 
   class BoundaryCondition; 

class SourceTermFactory
{
public:
  /**
   *  @brief obtain a reference to the SourceTermFactory.
   */
  static SourceTermFactory& self();

  /**
   *  @brief Register a source term on the specified transport equation.
   *
   *  @param eqnName The name of the transport equation to place the source term on.
   *  @param builder The SourceTermBuilder object to build the SourceTerm object.
   *
   *  SourceTermBuilder objects should be heap-allocated using "new".
   *  Memory management will be transfered to the SourceTermFactory.
   */
  void register_source_term( const std::string name,
                             SourceTermBase::Builder* builder );


  /**
   *  @brief Retrieve a vector of pointers to all SourceTerm
   *  objects that have been assigned to the transport equation with
   *  the specified name.
   *
   *  @param eqnName The name of the transport equation to retrieve
   *  SourceTerm objects for.
   *
   *  Note that this will construct new objects only as needed.
   */
  SourceTermBase& retrieve_source_term( const std::string name );

  struct SourceContainer{           ///< Hold the source names for this transport equation and the sign to either add or subtract from rhs.
    std::string name; 
    double      weight;             
  };

  void commonSrcProblemSetup( const ProblemSpecP& db ); 

  /** @brief Determine if a source term is contained in the factory. */
  bool source_term_exists( const std::string name );

  typedef std::map< std::string, SourceTermBase::Builder* > BuildMap;
  typedef std::map< std::string, SourceTermBase*    >       SourceMap;

  /** @brief Returns the list of all source terms in Map form. */ 
  SourceMap& retrieve_all_sources(){
    return _sources; }; 
 
  /** @brief Register all non-user defined sources */ 
  void registerSources( ArchesLabel* lab, const bool do_dmqom, const std::string which_dqmom );

  /** @brief Register all user-defined sources */ 
  void registerUDSources(ProblemSpecP& db, ArchesLabel* lab, BoundaryCondition* bcs, const ProcessorGroup* my_world);

  /** @brief Actually execute the sources */ 
  void sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep );

private:

  BuildMap  _builders;          ///< Builder map
  SourceMap   _sources;         ///< Sources map

  SourceTermFactory();
  ~SourceTermFactory();

  vector<SourceContainer> _active_sources;  ///< The list of all active source with associated weights.

}; // class SourceTermFactory
}  //Namespace Uintah

#endif
