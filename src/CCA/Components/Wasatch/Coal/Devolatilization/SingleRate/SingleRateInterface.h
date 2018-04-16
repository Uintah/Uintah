#ifndef DEV_SINGLERATE_INTERFACE_h
#define DEV_SINGLERATE_INTERFACE_h

/**
 *  \class   SingleRateInterface
 *  \ingroup Devolatilization - Single rate
 *  \brief   Provide an interface to connect Single Rate One-step model
 *           as a devolatiization model to coal models. 
 *
 *  \author  Babak Goshayeshi
 *  \date    May 2013 
 *
 *  \param   coalType     : the CoalType for this coal.
 *  \param   temperature  : particle temperature.
 *  \param   prtMassTag   : particle mass
 *  \param   pMass0Tag    : initial particle mass
 */

#include <map>
#include <expression/ExprLib.h>
#include <expression/Tag.h>
#include <expression/ExpressionID.h>

#include <CCA/Components/Wasatch/TimeStepper.h>
#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>

#include "SingleRateModel.h"


namespace DEV{ class DevolatilizationBase;}

namespace SNGRATE{
using WasatchCore::GraphCategories;
  

  /**
   *  \ingroup SNGRATE
   *  \class SingleRateInterface
   *  \brief Provides an interface to the Single Rate devolatilization model
   */
  template< typename FieldT >
  class SingleRateInterface: public DEV::DevolatilizationBase
  {
    GraphCategories& gc_;
    const Coal::CoalComposition coalComp_;
    const Expr::Tag pTempTag_, pMassTag_, pMass0Tag_;
    const bool isDAE_;
    SingleRateInformation singleRateData_;
    const Coal::StringNames& sNames_;
    Expr::TagList   mvCharTags_;
    bool haveRegisteredExprs_;

    /**
     *  \brief Parse ODEs relevant to the single rate model.
     */
    void parse_equations();

    /**
     *  \brief set tags that are not set in constructor initializer list
     */
    void set_tags();

    /**
     *  \brief Register all expressions required to implement the single rate model
     */
    void register_expressions();

    SingleRateInterface(); // no copying
    SingleRateInterface& operator=( const SingleRateInterface& ); // no assignment


  public:
    /**
     *  \param coalType  the CoalType for this coal.
     *  \param pTempTag  Particle temperature
     *  \param pMassTag  Particle mass
     *  \param pMass0Tag Initial particle mass
     */
    SingleRateInterface( GraphCategories& gc,
                         const Coal::CoalType& coalType,
                         const Expr::Tag pTempTag,
                         const Expr::Tag prtMassTag,
                         const Expr::Tag pMass0Tag,
                         const bool isDAE);


    /**
     *  \brief obtain the Tag for production rate of specified species
     */
    const Expr::Tag gas_species_src_tag( const DEV::DEVSpecies devspec ) const;

  };

} // namespace SNGRATE

#endif // DEV_SINGLERATE_INTERFACE_h
