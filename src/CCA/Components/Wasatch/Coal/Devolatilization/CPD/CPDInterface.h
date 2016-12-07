#ifndef CPDInterface_h
#define CPDInterface_h

/**
 *  \file CPDInterface.h
 *
 *  \defgroup CPD CPD Model
 */

#include <map>
#include <expression/ExprLib.h>
#include <expression/Tag.h>
#include <expression/ExpressionID.h>

#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>
#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

#include "CPDData.h"

namespace DEV { class DevolatilizationBase; }

namespace CPD{

using WasatchCore::GraphCategories;
using Coal::CoalType;

  /**
   *  \ingroup CPD
   *  \class CPDInterface
   *  \brief Provides an interface to the CPD model
   */
  template< typename FieldT >
  class CPDInterface: public DEV::DevolatilizationBase
  {
    const CPD::CPDInformation cpdInfo_;
    const Coal::CoalComposition& coalComp_;
    const double c0_, vMassFrac0_, tar0_;
    const Expr::Tag pTempTag_, pMassTag_, pMass0Tag_, kbTag_;
    const Coal::StringNames& sNames_;
    GraphCategories& gc_;

    Expr::Tag tarTag_, lbTag_, lb_rhsTag_, lbpTag_, lbp_rhsTag_;

    Expr::TagList g_rhsTags_, delta_rhsTags_, gTags_, deltaTags_, kgTags_;

    Expr::ExpressionID volatilesID_, l_rhsID_, crhsID_, g_rhsID_, delta_rhsID_, yiID_,
                       dyiID_, prd_rhsID_, char_rhsID_, tar_rhsID_, tarID_,
                       lbPopulation_rhsID_;

    Coal::CoalEquation *lbEqn_, *lbPopulationEqn_, *tarEqn_;
    Coal::CoalEqVec deltaEqns_, gEqns_, yEqns_;


    bool haveRegisteredExprs_;
    bool eqnsParsed_;

    /**
     *  \brief Parse ODEs relevant to the CPD model.
     */
    void parse_equations();

    /**
     *  \brief set tags that are not set in constructor initializer list
     */
    void set_tags();

    /**
     *  \brief Register all expressions required to implement the CPD model
     */
    void register_expressions();


    CPDInterface(); // no copying
    CPDInterface& operator=( const CPDInterface& ); // no assignment


  public:
    /**
     *  \param gc           reference to GraphCategories object
     *  \param coalType     the CoalType for this coal
     *  \param pTempTag     Particle temperature
     *  \param pMassTag     Particle mass
     *  \param pMass0Tag    Initial particle mass
     */
    CPDInterface( GraphCategories& gc,
                  const CoalType   coalType,
                  const Expr::Tag  pTempTag,
                  const Expr::Tag  pMassTag,
                  const Expr::Tag  pMass0Tag );

    /**
     *  \brief obtain a reference to the CPDInformation object.
     */
    const CPD::CPDInformation& get_cpd_info() const{ return cpdInfo_; }

    /**
     *  \brief obtain the Tag for production rate of specified species
     */
    const Expr::Tag gas_species_src_tag( const DEV::DEVSpecies devspec ) const;

    /**
     *  \brief obtain the Tag gi ( internal CPD calculation )
     */
    const Expr::TagList& gi_rhs_tags() const{ return g_rhsTags_; }

    /**
     *  \brief obtain the Tag for labile bridge consumption rate ( internal CPD tag )
     */
    const Expr::Tag& l_tag()       const{ return lbEqn_->solution_variable_tag(); }
    const Expr::Tag& l_rhs_tag()   const{ return lbEqn_->rhs_tag()              ; }
    const Expr::Tag& tar_tag()     const{ return tarEqn_->solution_variable_tag();}

  // const std::string get_species_name(int specindex);

  };

} // namespace CPD

#endif // CPDInterface_h
