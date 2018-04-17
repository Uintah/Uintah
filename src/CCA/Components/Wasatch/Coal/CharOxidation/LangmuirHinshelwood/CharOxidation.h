#ifndef CharOxidation_Expr_h
#define CharOxidation_Expr_h

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/c0_fun.h>
#include "CharOxidationFunctions.h"

namespace CHAR{


  /**
   *  \ingroup  CharOxidation
   *  \class    CharOxidation  ---   Shaddix equations [1]
   *  \author   Babak Goshayeshi (www.bgoshayeshi.com)
   *  \todo     improve documentation
   *
   *  \par Warnings:
   *   - Initial mass of Char must be coupled with CPD model
   *   - Initial amount of particle density must be coupled with CPD model
   *   - We must add a logic varible to the particle that shows if its
   *     eligible to enter to the reaction expression.
   *   - For structure parameter I have used initial amount of internal
   *     surface area.
   *  \param   prtDiamt      : Particle Diameter
   *  \param   tempPtag      : Particle Temperature
   *  \param   IntTempGas    : Interpolated gas temperature to the particle field
   *  \param   massFracO2Tag : Interpolated Mass Fraction of Oxygen to the particle field
   *  \param   totalMWTag    : Interpolated Total Molecular weight gas phase to the particle field
   *  \param   densityTag    : Density of Particle
   *  \param   charMassTag   : Char Mass
   *  \param   intGasPressTag: Interpolated Gas Pressure to the Particle Field
   *  \param   co2CORatioTag : CO2 over CO ratio production rate.
   *  \param   initprtmast   : Initial particle mass
   *  \param   chmodel       : Char oxidation model, LH and Fractal base on [1]
	 #                           and [2], respectively
   */
  template< typename FieldT >
  class CharOxidation
      : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS( FieldT, tempP_, tempG_, massFracO2_, totalMW_, density_, charMass_ )
    DECLARE_FIELDS( FieldT, gasPressure_, prtDiam_, initprtmas_ )

    const CharModel chmodel_;

    CharOxidation( const Expr::Tag& prtDiamt,
                   const Expr::Tag& tempPTag,
                   const Expr::Tag& intTempGas,
                   const Expr::Tag& massFracO2Tag,
                   const Expr::Tag& totalMWTag,
                   const Expr::Tag& densityTag,
                   const Expr::Tag& charMassTag,
                   const Expr::Tag& intGasPressTag,
                   const Expr::Tag& initprtmast,
                   const CharOxidationData oxiData,
                   const CharModel chmodel);

  public:
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag tempPTag_, tempGTag_, massFracO2Tag_, totalMWTag_, densityTag_, charMassTag_,
      intGasPressTag_, prtDiamt_, initprtmast_;
      CharOxidationData oxiData_;
      const CharModel chmodel_;
    public:
      Builder( const Expr::TagList& charOxTag,
               const Expr::Tag& prtDiamt,
               const Expr::Tag& tempPTag,
               const Expr::Tag& intTempGas,
               const Expr::Tag& massFracO2Tag,
               const Expr::Tag& totalMWTag,
               const Expr::Tag& densityTag,
               const Expr::Tag& charMassTag,
               const Expr::Tag& intGasPressTag,
               const Expr::Tag& initprtmast,
               const CharOxidationData oxiData,
               const CharModel chmodel);
      ~Builder(){}
      Expr::ExpressionBase* build() const  {
        return new CharOxidation<FieldT>(prtDiamt_ ,tempPTag_, tempGTag_, massFracO2Tag_,totalMWTag_,densityTag_, charMassTag_, intGasPressTag_, initprtmast_, oxiData_, chmodel_ );
      }
    };

    void evaluate();

    protected:
    CharOxidationData oxiData_;
  };



  // ###################################################################
  //
  //                          Implementation
  //
  // ###################################################################



  template< typename FieldT >
  CharOxidation<FieldT>::
  CharOxidation( const Expr::Tag& prtDiamt,
                 const Expr::Tag& tempPTag,
                 const Expr::Tag& intTempGas,
                 const Expr::Tag& massFracO2Tag,
                 const Expr::Tag& totalMWTag,
                 const Expr::Tag& densityTag,
                 const Expr::Tag& charMassTag,
                 const Expr::Tag& intGasPressTag,
                 const Expr::Tag& initprtmast,
                 const CharOxidationData oxiData,
                 const CharModel chmodel)
   : Expr::Expression<FieldT>(),
     chmodel_( chmodel ),
     oxiData_( oxiData )
  {
    tempP_       = this->template create_field_request<FieldT>( tempPTag       );
    tempG_       = this->template create_field_request<FieldT>( intTempGas     );
    massFracO2_  = this->template create_field_request<FieldT>( massFracO2Tag  );
    totalMW_     = this->template create_field_request<FieldT>( totalMWTag     );
    density_     = this->template create_field_request<FieldT>( densityTag     );
    charMass_    = this->template create_field_request<FieldT>( charMassTag    );
    gasPressure_ = this->template create_field_request<FieldT>( intGasPressTag );
    prtDiam_     = this->template create_field_request<FieldT>( prtDiamt       );
    initprtmas_  = this->template create_field_request<FieldT>( initprtmast    );
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void
  CharOxidation<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;
    // results[0]  :   Char Oxidation
    // results[1]  :   CO2/CO;

    typename Expr::Expression<FieldT>::ValVec& results = this->get_value_vec();
    double st1_, err_, st2_;
    const double R_ = 8.314;  // Gas constant
    const double f0err = 1e-15;
    const double h1err = 1e-10;
    double po2S, q, c_s, po2st,  cs;
    double f_0,f_1,f_2, po2S_2, po2S_r;
    double af,bf,cf,h1,h2,gama;
    double k1, k2,  co2CORatio, k_a_s;


    const double S0 = oxiData_.get_S_0(); // m2/kg same for all type of the coal -- Surfacearea
    const double Rpore = oxiData_.get_r_pore();  //  -- Surfacearea
    const double e0 = oxiData_.get_e0(); //  -- Surfacearea

    const double c0 = CPD::c0_fun( oxiData_.get_coal_composition().get_C(),
                                   oxiData_.get_coal_composition().get_O() );

    const double m0 = oxiData_.get_coal_composition().get_fixed_c()
                    + oxiData_.get_coal_composition().get_vm()*c0;

    const double Ocoal = oxiData_.get_O();
    typename FieldT::const_iterator iprtdiam    = prtDiam_    ->field_ref().begin();
    typename FieldT::const_iterator itempP      = tempP_      ->field_ref().begin();
    typename FieldT::const_iterator idensity    = density_    ->field_ref().begin();
    typename FieldT::const_iterator icharMass   = charMass_   ->field_ref().begin();
    typename FieldT::const_iterator iinitprtmas = initprtmas_ ->field_ref().begin();
    typename FieldT::const_iterator io2farc     = massFracO2_ ->field_ref().begin();
    typename FieldT::const_iterator itmw        = totalMW_    ->field_ref().begin();
    typename FieldT::const_iterator igaspres    = gasPressure_->field_ref().begin();
    typename FieldT::const_iterator itempg      = tempG_      ->field_ref().begin();

    typename FieldT::iterator iresult     = results[0]->begin();
    typename FieldT::iterator ico2coratio = results[1]->begin();

    const typename FieldT::iterator ire = results[0]->end();

    st2_ = S0/15.0 * *iinitprtmas *m0;
    for(; iresult != ire; ++iprtdiam, ++itempP, ++idensity, ++icharMass, ++iinitprtmas,
    ++io2farc, ++itmw, ++igaspres, ++itempg, ++iresult, ++ico2coratio ){

      if( *icharMass <= 0.0 ){
        *iresult = 0.0;
        continue;
      }

      /*
       ************    Start of Calculating particle surface area
       */
      //const double surfacearea = 3.141593* pow(*iprtdiam, 2.0); // Particle Surface Area
      const double initchar = *iinitprtmas;
      const double burnoff = ( initchar - *icharMass ) / (initchar);
      double surfacearea = S0 * (1.0 - burnoff) * initchar;

      if( Ocoal < 0.13 ){ // High Rank Coal
        const double particleV = 1.0 / (*idensity) * (*icharMass);
        // Loop to obtain the result for S which depends on the structure parameter.
        st1_ = st2_;
        st2_ = surfacearea - 1.0;
        err_ = 1.0;
        while (err_>1E-8) {
          st2_ = surfacearea * sqrt(1.0 - struc_param(*idensity,initial_particle_density(*iinitprtmas,*iprtdiam),
              e0, st1_ /particleV , Rpore)*log(1.0-burnoff));
          const double tmp = (st1_-st2_)/st1_;
          err_ = tmp*tmp;
          st1_ = st2_;
        }
        surfacearea = st2_;
      }

      /*
       ************    End of Calculating particle surface area
       */
      switch (chmodel_) {
        case FIRST_ORDER:
        case LH:
          /* The activation energies in [1] are said to be in kJ/mol, but if one calculates depletion
           * flux, q, assuming PO2_s = PO2_inf = 0.12 Atm and the maximum particle temperature (~2100K)
           * the result is ~10^-2 mol/m^2-s, and according to figure 12 it should be ~10 mol/m^2-s.
           * However, if the activation energies are assumed to be in J/mol q is calculated to be
           * ~10mol/m^2-s. Therefore it is assumed that the activation energies are in J/mol rather than
           * kJ/mol.
           *
           * Note, this also explains the nearly constant predicted rate of char consumption shown in
           * figure 14.
           */
          if( Ocoal < 0.13 ){ // High Rank Coal
            k1 = (61.0)*exp(-0.5/R_/ *itempP);
            k2 = (20.0)*exp(-107.4/R_/ *itempP);
          }
          else {
            k1 = (93.0)*exp(-0.1/R_/ *itempP);
            k2 = (26.2)*exp(-109.9/R_/ *itempP);
          }
          break;

        case FRACTAL:
          k_a_s = ka_fun(e0, S0, *idensity, *iprtdiam, *itempP) * surfacearea;
          break;

        case CCK:
        case INVALID_CHARMODEL:
          assert(false);
          break;
      }

      cs = (*io2farc * *itmw / (32.0) * *igaspres / R_ / *itempg);
      const double po2Inf = cs*(R_)* *itempg;
      c_s = *igaspres/R_/ *itempg;


      // for f1
      double po2S_1 = po2Inf;
      f_1 = SurfaceO2_error( q, co2CORatio, po2S_1, *igaspres, *itempg, *itempP, po2Inf, c_s, *iprtdiam, k1, k2, k_a_s, chmodel_);

      // for f2
      po2S_2 = 0.0;
      f_2 = SurfaceO2_error( q, co2CORatio, po2S_2, *igaspres, *itempg, *itempP, po2Inf, c_s, *iprtdiam, k1, k2, k_a_s, chmodel_);
      //f_2 = SurfaceO2_error(q, co2CORatio, po2s, *igaspres, *itempg, po2Inf, c_s);

      // for f0
      po2S = po2Inf/2.0;
      po2st = po2S + 100;
      f_0   = 1.0;

      bool dobisection = false;
      err_ =1.0;

      int i=1;
      while ( err_ > f0err ) {
        i++;
        if(i>100){
          dobisection = true;
          break;
        }
        po2st = po2S;

        // Function part
        // for f0
        f_0 = SurfaceO2_error( q, co2CORatio, po2S, *igaspres, *itempg, *itempP, po2Inf, c_s, *iprtdiam, k1, k2, k_a_s, chmodel_);
        err_ = f_0*f_0;
        h1 = po2S_1-po2S;
        h2 = po2S-po2S_2;
        if (h1 < h1err || err_ < f0err) {
          break;
        }
        gama = h2/h1;
        af = (gama*f_1-f_0*(1+gama)+f_2)/(gama*h1*h1*(1.0+gama));
        bf = (f_1-f_0-af*h1*h1)/h1;
        cf = f_0 ;

        if (bf > 0){
          po2S_r = po2S- 2.0*cf/(bf+sqrt(bf*bf-4.0*af*cf));
        }
        else {
          po2S_r = po2S- 2.0*cf/(bf-sqrt(bf*bf-4.0*af*cf));
        }
        if (po2S_r>po2S) {
          if (po2S_1-po2S_r < h1err){
            po2S_r = po2S_r - h1/10;
          }
          po2S_2 = po2S;
          f_2    = f_0;

          po2S   = po2S_r;
        }
        else {
          po2S_1 = po2S;
          f_1    = f_0;

          po2S   = po2S_r;
        }

        if (po2S > po2Inf + h1err || po2S < 0){
          dobisection = true;
          break;
        }
      }

      /*
       ************    Start of Bisection Method
       */

      // Initialization
      double  f_2;
      double bisec=0.0;

      if( dobisection ){

        h1 = po2Inf/10000;
        f_1 = SurfaceO2_error( q, co2CORatio, h1, *igaspres, *itempg, *itempP, po2Inf, c_s, *iprtdiam, k1, k2, k_a_s, chmodel_);

        h2 = po2Inf;
        f_2 = SurfaceO2_error( q, co2CORatio, h2, *igaspres, *itempg, *itempP, po2Inf, c_s, *iprtdiam, k1, k2, k_a_s, chmodel_);
        if (f_1 * f_2 > 0.0) {
          bisec = f_1 + (f_2-f_1)/2.0;
        }
      }
      err_ = 1.0;
      i=0;
      double halff = 1.0;
      while( dobisection && err_>f0err && halff/h2>f0err ){
        i++;
        halff = (h2-h1)/2.0;
        po2S = h1 + halff ;
        err_ = SurfaceO2_error( q, co2CORatio, po2S, *igaspres, *itempg, *itempP, po2Inf, c_s, *iprtdiam, k1, k2, k_a_s, chmodel_);
        if (err_ > bisec) {
          h2=po2S;
        }
        else {
          h1=po2S;
        }

        err_ *= err_;

        if( i>100 ){
          proc0cout << " CharOxidation.h : bisection method couldn't converge on oxygen concentration at particle surface! \n";
          break;
        }
      }
      /*
       ************    End of Bisection Method
       */
      *ico2coratio = co2CORatio;
      *iresult = -q * 12.0 * 3.141593* pow(*iprtdiam, 2.0) / 1000.0;

    }
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  CharOxidation<FieldT>::
  Builder::Builder( const Expr::TagList& charOxTag,
                    const Expr::Tag& prtDiamt,
                    const Expr::Tag& tempPTag,
                    const Expr::Tag& intTempGas,
                    const Expr::Tag& massFracO2Tag,
                    const Expr::Tag& totalMWTag,
                    const Expr::Tag& densityTag,
                    const Expr::Tag& charMassTag,
                    const Expr::Tag& intGasPressTag,
                    const Expr::Tag& initprtmast,
                    const CharOxidationData oxiData,
                    const CharModel chmodel)
  : ExpressionBuilder(charOxTag),
    tempPTag_      ( tempPTag ),
    tempGTag_      ( intTempGas ),
    massFracO2Tag_ ( massFracO2Tag ),
    totalMWTag_    ( totalMWTag ),
    densityTag_    ( densityTag ),
    charMassTag_   ( charMassTag ),
    intGasPressTag_( intGasPressTag ),
    prtDiamt_      ( prtDiamt ),
    initprtmast_   ( initprtmast ),
    oxiData_( oxiData ),
    chmodel_( chmodel )
  {
    if( chmodel == INVALID_CHARMODEL ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << "\nUnsupported char model\n";
      throw std::invalid_argument( msg.str() );
    }

    if( chmodel == CCK ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__
          << "\nCCK model should not be used with CharOxidation object"
          << "\nas it already considers char oxidation.";
      throw std::runtime_error( msg.str() );
    }
  }

  //--------------------------------------------------------------------
} // namespace CHAR

/*
 [1] "Combustion kinetics of coal chars in oxygen-enriched environments", Jeffrey J. Murphy,Christopher R. Shaddix, Combustion and Flame 144 (2006) 710-729
 [2] He, Wei, Yuting Liu, Rong He, Takamasa Ito, Toshiyuki Suda, Toshiro Fujimori, Hideto Ikeda, and Jun’ichi Sato. 
     "Combustion Rate for Char with Fractal Pore Characteristics" Combustion Science and Technology no. July (July 11, 2013)
 */

#endif // CharOxidation_Expr_h
