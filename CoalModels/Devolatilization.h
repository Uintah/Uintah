#ifndef Uintah_Component_Arches_Devolatilization_h
#define Uintah_Component_Arches_Devolatilization_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    Devolatilization
  * @author   Charles Reid
  * @date     October 2009
  *
  * @brief    A devolatilization model parent class 
  *
  */

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

//class ArchesLabel;
//class DevolatilizationBuilder: public ModelBuilder
//{
//public: 
//  DevolatilizationBuilder( const std::string          & modelName,
//                           const vector<std::string>  & reqICLabelNames,
//                           const vector<std::string>  & reqScalarLabelNames,
//                           const ArchesLabel          * fieldLabels,
//                           SimulationStateP           & sharedState,
//                           int qn );
//
//  ~DevolatilizationBuilder(); 
//
//  // don't declare build() function as virtual, this will create virtual "thunk"-ing problems
//
//private:
//
//}; 

// End Builder
//---------------------------------------------------------------------------

class Devolatilization: public ModelBase {
public: 

  Devolatilization( std::string modelName, 
                         SimulationStateP& shared_state, 
                         const ArchesLabel* fieldLabels,
                         vector<std::string> reqICLabelNames, 
                         vector<std::string> reqScalarLabelNames,
                         int qn );

  virtual ~Devolatilization();

  /** @brief  Grab model-independent devolatilization parameters */
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief  Get raw coal reaction rate */
  double calcRawCoalReactionRate();

  /** @brief  Get gas volatile production rate */
  double calcGasDevolRate();

  /** @brief  Get char production rate */
  double calcCharProductionRate();

  inline static const modelTypeEnum getTypeDescription() {
    return DEVOLATILIZATION; };

protected:

  double d_lowModelClip; 
  double d_highModelClip; 

  double d_w_scaling_factor; 
  double d_w_small; // "small" clip value for zero weights

}; // end Devolatilization
} // end namespace Uintah
#endif
