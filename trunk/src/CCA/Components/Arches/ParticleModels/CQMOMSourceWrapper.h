#ifndef Uintah_Component_Arches_CQMOMSourceWrapper_h
#define Uintah_Component_Arches_CQMOMSourceWrapper_h
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>

//==========================================================================

/**
 * @class CQMOMSourceWrapper
 * @author Alex Abboud
 * @date August 2014, revised June 2015
 *
 * @brief Construct the source terms for moment equations in CQMOM, the moment forumlation the growth term for
 *        $\f m_{i,j,k} $\f of the $\f 1^{st} \f$ internal coordiante appears as $\f \int \int \int d/d_{r_1} G(r_1)
 *        \eta(r_1,r_2,r_3) r_1^i r_2^j r_3^k dr_1 dr_2 dr_3 \f$ using integration by parts this can be simplified to
 *        $\f i \int \int \int G(r_1) \eta(r_1,r_2,r_3) r_1^{i-1} r_2^j r_3^k \f$, then the quadrature approximation can be applied
 *        such that the integral can be expressed as a summation of the nodes as
 *        \f$ i \sum_\alpha^N \omega_k r_{1,\alpha}^{i-1} r_{2,\alpha}^j r_{3,\alpha}^k \f$
 *        This forumla only holds if G(r_1) is of form dr_1/dt
 *
 *        The refactor of this source term wrapper will be scheduled by explicit solver and not the CQMOM eqns
 *        Only one soruce term object will exist, and it will calculate all of the source terms for every moment eqn
 */

namespace Uintah{

  class ArchesLabel;
  class CQMOMSourceWrapper {

  public:

    CQMOMSourceWrapper( ArchesLabel* fieldLabels );

    ~CQMOMSourceWrapper();

    typedef std::vector<int> MomentVector;

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
                          DataWarehouse* new_dw, 
                          const int timeSubstep);

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

    /** @brief Return the bool for if soruce terms are on. */
    inline bool getAddSources(){ return d_addSources; };

  private:

    ArchesLabel* d_fieldLabels;

    bool d_addSources;                          //bool if source terms found

    std::vector<int> N_i;                       //vector of number of quadrature nodes in each dimension
    std::vector<int> nIC;                       //vector of indexes for source term IC
    std::vector<MomentVector> momentIndexes;    //List of all moment indexes

    int M;                                      //number of internal coordiantes
    int _N;                                     //total number of nodes
    int nMoments;                               //total number of moments
    int nSources;                               //total number of source terms

    std::vector<const VarLabel*> d_sourceLabels;  //labels for all source terms for all the moments
    const VarLabel* volfrac_label;                //volfrac label
    std::vector<const VarLabel *> d_nodeSources;  //list of all node source for all models

  }; // class CQMOMWrapper
} // namespace Uintah

#endif
