#ifndef Uintah_Component_Arches_MultiPointConst_h
#define Uintah_Component_Arches_MultiPointConst_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/Box.h>

namespace Uintah{

//---------------------------------------------------------------------------
// Builder
class MultiPointConstBuilder: public SourceTermBuilder
{
public: 
  MultiPointConstBuilder(std::string srcName, 
                      vector<std::string> reqLabelNames, 
                      SimulationStateP& sharedState);
  ~MultiPointConstBuilder(); 

  SourceTermBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

/** @class    MultiPointConst
  * @atuhor   Jeremy Thornock
  * @date     July 2010
  * 
  * @brief    Source term for injecting constant sources at specified regions in the flow via geom_objects.
  *
  */

class MultiPointConst: public SourceTermBase {
public: 

  MultiPointConst( std::string srcName, SimulationStateP& shared_state, 
                vector<std::string> reqLabelNames );

  ~MultiPointConst();

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

  inline string getType() {
    return "MultiPointConst";
  };

private:

  double d_constant; 
  std::vector<GeometryPieceP> d_geomPieces; 

}; // end MultiPointConst
} // end namespace Uintah
#endif
