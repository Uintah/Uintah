#ifndef Uintah_Component_Arches_CQMOMSourceWrapper_h
#define Uintah_Component_Arches_CQMOMSourceWrapper_h
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/TransportEqns/Convection_CQMOM.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>

//==========================================================================

/**
 * @class CQMOMSourceWrapper
 * @author Alex Abboud
 * @date August 2014
 *
 * @brief Construct the source terms for moment equations in CQMOM, the moment forumlation the growth term for
 *        $\f m_{i,j,k} $\f of the $\f 1^{st} \f$ internal coordiante appears as $\f \int \int \int d/d_{r_1} G(r_1)
 *        \eta(r_1,r_2,r_3) r_1^i r_2^j r_3^k dr_1 dr_2 dr_3 \f$ using integration by parts this can be simplified to
 *        $\f i \int \int \int G(r_1) \eta(r_1,r_2,r_3) r_1^{i-1} r_2^j r_3^k \f$, then the quadrature approximation can be applied
 *        such that the integral can be expressed as a summation of the nodes as
 *        \f$ i \sum_\alpha^N \omega_k r_{1,\alpha}^{i-1} r_{2,\alpha}^j r_{3,\alpha}^k \f$
 *        This forumla only holds if G(r_1) is of form dr_1/dt
 *
 *
 */

namespace Uintah{
  
  class ArchesLabel;
  class CQMOMSourceWrapper {

  public:
    
    CQMOMSourceWrapper( ArchesLabel* fieldLabels, std::string sourceName, std::vector<int> momentIndex, int nIC );
    
    ~CQMOMSourceWrapper();
    
    /** @brief return instance of this */
    static CQMOMSourceWrapper& self();
    
    /** @brief Set any parameters from input file, initialize any constants, etc.. */
    void problemSetup(const ProblemSpecP& inputdb);
    
    /** @brief schedule creation of the source term for each moment equation */
    void sched_buildSourceTerm( const LevelP& level,
                                SchedulerP& sched, int timeSubStep );
    
    /** @brief Actual creation of the source term for each moment equation  */
    void buildSourceTerm( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);
    
    /** @brief Schedule the initialization of the variables */
    void sched_initializeVariables( const LevelP& level, SchedulerP& sched );
    
    /** @brief Actually initialize the variables at the begining of a time step */
    void initializeVariables( const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );
    
    // --------------------------------------
    // Access functions:
    
    /** @brief Return the VarLabel for this equation's source term. */
    inline const VarLabel* getSourceLabel(){ return d_modelLabel; };
    
    /** @brief return the moment index vector of this equation */
    inline const std::vector<int> getMomentIndex(){ return d_momentIndex; };
    
  private:
    
    ArchesLabel* d_fieldLabels;
    const VarLabel * d_modelLabel;        //var label for the source of this model
    std::vector<int> d_momentIndex;        // moment index for this transport equation, needed for convective and source closure
    std::vector<int> N_i;                // vector of number of quadrature nodes in each dimension
    
    const VarLabel * d_momentLabel;       //Label for the moment of this transport equation
    int M;                               //number of internal coordiantes
    int _N;                              //total number of nodes
    int _nIC;
    
    std::string sourceName;
    
    std::vector<const VarLabel *> d_nodeSources;
    std::string model_name;
    struct constCCVarWrapper {
      constCCVariable<double> data;
    };
    
  }; // class CQMOMEqn
} // namespace Uintah

#endif
