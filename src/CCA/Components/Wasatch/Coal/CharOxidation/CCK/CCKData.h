#ifndef CCKData_h
#define CCKData_h
#include "CCKFunctions.h"
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>

namespace CCK{

  const double calc_gamma( const double& pTemp);

  /**
   *  \class    CCKData
   *  \author   Josh McConnell
   *  \date     June 2015
   */

  class CCKData{

  public:
    CCKData( const Coal::CoalType coalType );

    /**
     * \brief Returns value for tau/f
     */
    inline double get_tau_f() const {return tau_f_;}

    /**
     * \brief Returns porosity of char
     */
    inline double get_char_porosity() const{ return epsilon_;}

    /**
     * \brief Returns mode-of-burning parameter
     */
    inline double get_mode_of_burning_param() const{ return modeOfBurningParam_; }

    /**
     * \brief Returns minimum ash thickness
     */
    inline double get_min_ash_thickness() const{ return deltaMin_; }

    /**
     * \brief Returns minimum ash porosity
     */
    inline double get_min_ash_porosity() const{ return thetaMin_; }

    /**
     * \brief Returns ash density
     */
    inline double get_nonporous_ash_density() const{ return ashDensity_; }

    /**
     * \brief Returns object containing coal composition
     */
    Coal::CoalComposition get_coal_comp() const{ return charData_.get_coal_composition(); }

    /**
     * \brief Returns oxygen mass fraction of coal
     */
    double get_O() const{ return o_;}

    /**
     * \brief Returns carbon fraction of coal
     */
    double get_C() const{ return c_;}

    /**
     * \brief Returns fixed carbon mass fraction of coal
     */
    double get_fixed_C() const{ return fixedCarbon_;}

    /**
     * \brief Returns volatile mass fraction of coal
     */
    double get_vm() const{ return volatiles_;}

    /**
     *  \brief returns molecular weight of a given species
     */
    const double get_mw( CHAR::CharGasSpecies spec ) const { return charData_.get_mw( spec ); };

    /**
     * \brief Returns ash mass fraction of coal
     */
    double get_ash() const{ return xA_;}

    /**
     * \brief Returns frequency factor for thermal annealing model
     */
    double get_aD() const{ return aD_;}

    /**
     * \brief Returns standard deviation of log(Activation energy)
     *        for thermal annealing model
     */
    double get_neD_std_dev() const{ return stdDev_neD_;}

    /**
     * \brief Returns mean of log(Activation energy)
     *        for thermal annealing model
     */
    double get_neD_mean() const{ return mean_neD_;}

    /**
     * \brief Returns stuctural parameter for random pore model
     */
    double get_struct_param() const { return 4.6;}


    /**
     * \brief Returns a vector of activation energies used for
     *        calculation of annealing factor in for thermal
     *        annealing model
     */
    CHAR::Vec get_eD_vec() const{ return eDVec_; }

    /**
     * \brief Returns CO2-CO ratio
     */
    double co2_co_ratio( const double pTemp,
                         const double csO2 ) const;

    /**
     * \brief Returns a mass transfer coefficient
     */

    double massTransCoeff( const double& nSh,
                           const double& Deff_i,
                           const double& theta,
                           const double& prtDiam,
                           const double& coreDiam,
                           const double& mTemp,
                           const double& nu ) const;

    /**
     * \brief Returns a vector of forward rate constants for reactions
     *        considered within the CCK model
     */
    CHAR::Vec forward_rate_constants( const double& pressure,
                                      const double& pTemp,
                                      const double& saFactor ) const;

    /**
     * \brief Returns a vector of reverse rate constants for reactions
     *        considered within the CCK model
     */
    CHAR::Vec reverse_rate_constants( const double& pTemp,
                                      const double& saFactor) const;

    /**
     * \brief Returns a vector of species consumption rates calculated
     *        using the CCK model
     */
    CHAR::Vec char_consumption_rates( const CHAR::Array2D& binDCoeff,
                                    const CHAR::Vec&     x,
                                    const CHAR::Vec&     ppSurf,
                                    const CHAR::Vec&     ppInf,
                                    const CHAR::Vec&     kFwd,
                                    const CHAR::Vec&     kRev,
                                    const double&  pressure,
                                    const double&  prtDiam,
                                    const double&  prtTemp,
                                    const double&  mTemp,
                                    const double&  coreDiam,
                                    const double&  coreDens,
                                    const double&  theta,
                                    const double&  epsilon_,
                                    const double&  tau_f_,
                                    const double&  nSh,
                                    const double&  gamma,
                                    double&        CO2_CO_ratio,
                                    CHAR::Vec&     pError ) const;


  protected:
    double tau_f_, epsilon_, modeOfBurningParam_,
    a0_over_a2_, a1_over_a2_,
    deltaMin_, thetaMin_, ashDensity_,
    percentC_, o_, c_, fixedCarbon_, volatiles_, xA_,
    mean_neD_, aD_, stdDev_neD_;

    CHAR::Vec diffVol_;
    CHAR::Vec aFwd0_, aRev0_, eaFwd_, eaRev_, eDVec_;

  private:

    CHAR::CharOxidationData charData_;
  };
}// namespace CCK


#endif /* CCKData_h */
