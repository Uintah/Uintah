/*
 * The MIT License
 *
 * Copyright (c) 2015-2017 The University of Utah
 *
  * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CCKModel_Expr_h
#define CCKModel_Expr_h

#include <cmath>
#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/c0_fun.h>
#include "CCKData.h"
#include "SolveLinearSystem.h"

namespace CCK{


void check_for_errors( const size_t& numIter, CHAR::Vec pSurf, CHAR::Vec pBulk,
                       CHAR::Vec pError, const double& maxError,
                       const size_t& i_maxError )
{
  if( numIter > 100 ){
    std::ostringstream msg;

    msg << "\nCalculated surface and bulk pressures, and errors"
        << "\n[CO2, CO O2 H2 H2O CH4]\n"
        << "i\tpSurf\t\tpBulk\t\terror\n";
    for(size_t i = 0; i<pBulk.size(); ++i){
      msg << i << "\t"
          << pSurf[i] << "\t\t"
          << pBulk[i] << "\t\t"
          << pError[i]
          << std::endl;
    }
    msg << "\nmax Error: " << maxError
        << "\n species index: " << i_maxError
        << std::endl
        << __FILE__ << " : " << __LINE__ << std::endl
        << "Newton solve failed to converge after 100 iterations.\n\n";
    throw std::runtime_error( msg.str() );
  }
}

/**
 *  \class  CCKodel
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates char oxidation and gasification by CO2, H2O, and H2
 *         using the CCK model
 */

template< typename FieldT >
class CCKModel
 : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS( FieldT, massFracs_  )
  DECLARE_VECTOR_OF_FIELDS( FieldT, gasProps_   )
  DECLARE_VECTOR_OF_FIELDS( FieldT, prtProps_   )

  const CCKData&  cckData_;
  const double epsilon_, tau_f_, gasConst_, shNo_;
  CHAR::Vec mwVec_;

  std::vector<typename FieldT::const_iterator> yIVec_;
  std::vector<typename FieldT::const_iterator> resultIVec_;
  
  CCKModel( const Expr::TagList& massFracTags,
            const Expr::TagList& gasPropTags,
            const Expr::TagList& prtPropTags,
            const CCKData&       cckData );

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a CCKModel expression
     *  @param resultTags the tag for the value that this expression computes
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::TagList& massFracTags,
             const Expr::TagList& gasPropTags,
             const Expr::TagList& prtPropTags,
             const CCKData&       cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTags, nghost ),
        massFracTags_( massFracTags ),
        gasPropTags_ ( gasPropTags  ),
        prtPropTags_ ( prtPropTags  ),
        cckData_     ( cckData      )
    {}

    Expr::ExpressionBase* build() const{
      return new CCKModel<FieldT>( massFracTags_, gasPropTags_, prtPropTags_,
                                   cckData_ );
    }

  private:
    const Expr::TagList massFracTags_, gasPropTags_, prtPropTags_;
    const CCKData&  cckData_;
  };

  ~CCKModel(){};
  void evaluate();
};


// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
CCKModel<FieldT>::CCKModel( const Expr::TagList& massFracTags,
                            const Expr::TagList& gasPropTags,
                            const Expr::TagList& prtPropTags,
                            const CCKData&       cckData )
  : Expr::Expression<FieldT>(),
    cckData_ ( cckData   ),
    epsilon_ ( cckData.get_char_porosity() ),
    tau_f_   ( cckData.get_tau_f() ),
    gasConst_( 8.3144621 ),
    shNo_    ( 2.0       ) // set Sherwood number to 2
{
  mwVec_.clear();
  mwVec_.push_back( cckData_.get_mw(CHAR::CO2) );
  mwVec_.push_back( cckData_.get_mw(CHAR::CO ) );
  mwVec_.push_back( cckData_.get_mw(CHAR::O2 ) );
  mwVec_.push_back( cckData_.get_mw(CHAR::H2 ) );
  mwVec_.push_back( cckData_.get_mw(CHAR::H2O) );
  mwVec_.push_back( cckData_.get_mw(CHAR::CH4) );

  this->template create_field_vector_request<FieldT>( massFracTags, massFracs_ );
  this->template create_field_vector_request<FieldT>( gasPropTags,  gasProps_  );
  this->template create_field_vector_request<FieldT>( prtPropTags,  prtProps_  );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
CCKModel<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typedef typename FieldT::const_iterator FieldIter;
  typename Expr::Expression<FieldT>::ValVec&  result = this->get_value_vec();

  /*
   * There will be 5 results (in this order) for the following species:
   *
   * ***************     [ CO2, CO, O2, H2, H2O ]     ****************
   *
   * Char consumption rates (rC) are in (kg char consumed)/sec
   */
  typename FieldT::iterator irC_CO2    = result[0]->begin();
  typename FieldT::iterator irC_CO     = result[1]->begin();
  typename FieldT::iterator irC_O2     = result[2]->begin();
  typename FieldT::iterator irC_H2     = result[3]->begin();
  typename FieldT::iterator irC_H2O    = result[4]->begin();

  //ratio of CO2 to CO for char oxidation
  typename FieldT::iterator iCO2_CO_ratio = result[5]->begin();


  // initial mass fraction of char in volatiles
  const double c0 = CPD::c0_fun( cckData_.get_C(), cckData_.get_O() );

  // char contribution from volatiles (mass fraction)
  const double charMassFrac0 = cckData_.get_fixed_C() + cckData_.get_vm()*c0;

  // pack a vector with mass fraction iterators.
  yIVec_.clear();
  for( size_t i=0; i<massFracs_.size(); ++i )
  {
    yIVec_.push_back( massFracs_[i]->field_ref().begin() );
  }

  FieldIter igasTemp  = gasProps_[0]->field_ref().begin(); // gas temperature
  FieldIter ipressure = gasProps_[1]->field_ref().begin(); // pressure
  FieldIter imixMW    = gasProps_[2]->field_ref().begin(); // mixture molecular weight

  FieldIter iprtTemp  = prtProps_[ 0]->field_ref().begin(); // particle temperature
  FieldIter ipMass0   = prtProps_[ 1]->field_ref().begin(); // initial particle mass
  FieldIter ipMass    = prtProps_[ 2]->field_ref().begin(); // particle mass
  FieldIter icharMass = prtProps_[ 3]->field_ref().begin(); // char mass
  FieldIter icharConv = prtProps_[ 4]->field_ref().begin(); // char mass
  FieldIter icoreDens = prtProps_[ 5]->field_ref().begin(); // core density
  FieldIter icoreDiam = prtProps_[ 6]->field_ref().begin(); // core diameter
  FieldIter iprtDiam  = prtProps_[ 7]->field_ref().begin(); // particle diameter
  FieldIter itheta    = prtProps_[ 8]->field_ref().begin(); // ash film porosity
  FieldIter idelta    = prtProps_[ 9]->field_ref().begin(); // ash film thickness
  FieldIter iaFactor  = prtProps_[10]->field_ref().begin(); // thermal annealing deactivation factor

  const typename FieldT::const_iterator iEnd = prtProps_[0]->field_ref().end();

  // loop over each particle
  for(; iprtTemp != iEnd; ++igasTemp, ++ipressure, ++imixMW, ++iprtTemp, ++ipMass0,
        ++ipMass, ++icharMass, ++icharConv, ++icoreDens, ++icoreDiam, ++iprtDiam, ++itheta,
        ++idelta, ++iaFactor,
        ++irC_CO2, ++irC_CO, ++irC_O2, ++irC_H2, ++irC_H2O, ++iCO2_CO_ratio){


    // set parameters that are constant within this for-loop
    const double initCharMass   = charMassFrac0 * *ipMass0;
    // jcs it appears that charConversion isn't used.  That implies that charMass and initCharMass are also unused...
    const double charConversion = 1.0 - *icharMass/initCharMass;
    const double mTemp = 0.5*( *iprtTemp + *igasTemp );

    CHAR::Vec xInf;  xInf.clear();  // bulk species mole fractions
    CHAR::Vec pBulk; pBulk.clear(); // bulk species partial pressures

    CHAR::Vec::const_iterator imw = mwVec_.begin();
    // Assemble vectors for bulk mole fractions and partial pressures;
    for( typename std::vector<typename FieldT::const_iterator>::iterator iy = yIVec_.begin();
        iy!=yIVec_.end(); ++iy, ++imw )
    {

      const double moleFrac = **iy * *imixMW / *imw;
      xInf.push_back( moleFrac );
      pBulk.push_back( *ipressure * moleFrac );

      ++(*iy); // increment iterators for mass fractions to the next particle
    }

    size_t n = pBulk.size();

    // Construct array of binary diffusion coefficients
    CHAR::Array2D dFick = CHAR::binary_diff_coeffs( *ipressure, mTemp );

//============================================================================//
//============================================================================//
    /* Get rate constants from correlations, thermal annealing deactivation
     * factor (*iaFactor), and surface area reduction factor (fRPM) from
     * random pore model.
     */
    const double fRPM = (1 - *icharConv)*sqrt(
                         1 - cckData_.get_struct_param()*log(1 - *icharConv));

    const double saFactor = fRPM*(*iaFactor);

    const CHAR::Vec kFwd = cckData_.forward_rate_constants( *ipressure, *iprtTemp, saFactor );
    const CHAR::Vec kRev = cckData_.reverse_rate_constants( *iprtTemp, saFactor );

    const double gamma = calc_gamma( *iprtTemp );

    /* Calculate partial pressures of inert components. Calculations herein
     * will assume the only inert component at particle surfaces is N2
     * (if at all). It is also assumed that the inert component has the same
     * partial pressure at the particle surface and the bulk.
     */
    const double xSum    = std::accumulate( xInf.begin(), xInf.end(), 0.0 );
    // jcs it appears that ppInert is unused, which means that xSum is also unused.
    const double ppInert = *ipressure * fmin(1.0, fmax(0.0, 1.0 - xSum ) );
    double       CO2_CO_ratio;

     /* Get map determining which species will be solved for. If a species is
      * neither consumed nor produced, its index will not be an element of this
      * map.
      */
     const std::map<size_t, size_t> indexMap = speciesSolveMap(pBulk);
     size_t m  = indexMap.size();

     CHAR::Vec pSurf1, pSurf2, pStep;
     pStep.assign  (m, 0.0);

     bool isInvertable;

     CHAR::Array2D J( boost::extents[m][m] );

     pSurf1.resize(n);
     for(size_t i = 0; i<n; ++i){ pSurf1[i] = pBulk[i];}

     //Variables that will be set by cckData_.charConsumptionRates
     CHAR::Vec error1;  error1.assign(n, 0.0);
     CHAR::Vec error2 = error1;
     CHAR::Vec error3;  error3.assign(m, 0.0);
     CHAR::Vec error_old;

     CHAR::Vec rC; rC.clear();
     CHAR::Vec dPvec; dPvec.resize(n);

     double tol = 1.0e-5;
     double maxError = tol*10;
     size_t i_maxError = 0;
     size_t count = 0;

     // Do surface pressure calculations if reacting species are present
     if( m > 0){

/******************************************************************************/
//*************** begin while-loop for surface rxn calculations **************//
/******************************************************************************/
     while (maxError > tol){
       for( size_t i = 0; i<n; ++i){
         dPvec[i] = fmax( pSurf1[i]*1.0e-6, 1.0e-6 );
       }
       error_old = error3;
       ++count;

       check_for_errors( count, pSurf1, pBulk, error1, maxError, i_maxError );

       rC = cckData_.char_consumption_rates
            ( dFick,      xInf,      pSurf1,   pBulk, kFwd,       kRev,
              *ipressure, *iprtDiam, *iprtTemp, mTemp, *icoreDiam, *icoreDens,
              *itheta,    epsilon_,  tau_f_,    shNo_, gamma,      CO2_CO_ratio,
              error1 );

       for ( std::map<size_t, size_t>::const_iterator im1 = indexMap.begin();
             im1 != indexMap.end(); ++im1){
         size_t index1 =  im1->second;

         pSurf2 = pSurf1;
         pSurf2[index1] = pSurf1[index1] + dPvec[index1];

         rC = cckData_.char_consumption_rates
              ( dFick,      xInf,      pSurf2,   pBulk, kFwd,       kRev,
                *ipressure, *iprtDiam, *iprtTemp, mTemp, *icoreDiam, *icoreDens,
                *itheta,    epsilon_,  tau_f_,    shNo_, gamma,      CO2_CO_ratio,
                error2 );

         error3[im1->first] = error1[index1];

         /* construct Jacobian matrix. NOTE: the entries of this matrix are -d(error_i)/d(pSurf_j)
          * rather than d(error_i)/d(pSurf_j) so that [J]*pStep = pSurf may be solved for pStep,
          * where pSurf_new = pSurf_old + pStep.
          */
         for ( std::map<size_t, size_t>::const_iterator im2 = indexMap.begin();
               im2 != indexMap.end(); ++im2){
           size_t index2 = im2->second;

           J[im2->first][im1->first] = (error1[index2] - error2[index2])/dPvec[index2];
         }//for
       }//for

       isInvertable = SolveLinearSystem(J, error3, pStep );
       if(isInvertable == false){
         std::ostringstream msg;
         msg << __FILE__ << " : " << __LINE__ << std::endl
             << "CCK: Jacobian matrix used for surface composition solve is not invertible." << std::endl
             << std::endl;
         throw std::runtime_error( msg.str() );
       }

       maxError = 0;
       for ( std::map<size_t, size_t>::const_iterator im = indexMap.begin();
           im != indexMap.end(); ++im){
           const double p = pSurf1[im->second] + pStep[im->first];

           if(p<0){ pSurf1[im->second] = pSurf1[im->second]/2.0;}
           else   { pSurf1[im->second] = p;}

           double specError;
           i_maxError = 0;

           /* If the surface partial pressure of a species is less than 1e-2 Pa use the absolute
            * error. Otherwise, use the relative error. This is done because convergence issues
            * arise when the calculated partial pressures are very small.
            */
           specError = pSurf1[im->second] > 1e-2?
                       error3[im->first]/pSurf1[im->second] :
                       error3[im->first];

           if( maxError < fabs(specError) ){
            maxError   = fabs(specError);
            i_maxError = im->first;
           }//if
        }//for
     }//while
//--------------------------------------------------------------------
//--------------------------------------------------------------------

     // results are given in (kg char consumed)/s
      *irC_CO2 = rC[0];
      *irC_CO  = rC[1];
      *irC_O2  = rC[2];
      *irC_H2  = rC[3];
      *irC_H2O = rC[4];

      *iCO2_CO_ratio = CO2_CO_ratio;
     }

     /* If no reacting species are present, set reaction rates to zero
      */
     else{
       *irC_CO2 = 0;
       *irC_CO  = 0;
       *irC_O2  = 0;
       *irC_H2  = 0;
       *irC_H2O = 0;

       *iCO2_CO_ratio = cckData_.co2_co_ratio(*iprtTemp, 0);
     }
//--------------------------------------------------------------------
//--------------------------------------------------------------------
  }
}

//--------------------------------------------------------------------

}//namespace CCK

/*
 * [1]  Randy C. Shurtz. Effects of Pressure on the Properties of Coal Char Under
 *      Gasification Conditions at High Initial Heating Rates. (2011). All Theses
 *      and Dissertations. Paper 2877. http://scholarsarchive.byu.edu/etd/2877/
 *
 * [2] Stephen Niksa et. al. Coal conversion submodels for design applications at
 *     elevated pressures. Part I. devolatilization and char oxidation. Progress
 *     in Energy and Combustion Science 29 (2003) 425–477.
 *     http://www.sciencedirect.com/science/article/pii/S0360128503000339
 */
#endif // CCKModel_Expr_h
