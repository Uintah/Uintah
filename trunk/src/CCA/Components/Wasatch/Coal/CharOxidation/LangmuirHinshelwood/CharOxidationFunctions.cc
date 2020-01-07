#include "CharOxidationFunctions.h"

namespace CHAR{

  // -----------------------------------------------------------------------------
  double struc_param( const double rho,
                      const double rho_0,
                      const double e0,
                      const double s,
                      const double r )

  {
    // Modified Structure Parameter from Fei et al (2011).
    const double rhop = rho*(1.0-e0)/rho_0;
    const double psi = 2.0*rhop/s/r;
    return psi;
  }

  // -----------------------------------------------------------------------------
  double Do2_mix( const double& pressure,
                  const double& temp,
                  const double& co_ratio )
  {
    const double Cst= 1.013e-2;
    const double c13 = 1.0/3.0;
    // Diffusion of Oxygen in CO2
    const double D_CO2 = Cst * pow(temp,1.75)*pow((32.0+44.0)/(32.0*44.0),0.5)/
        (pressure * pow(pow(16.3, c13)+pow(26.7, c13),2.0)); // Danner and Daubert (1983)

    // Diffusion of Oxygen in CO
    const double D_CO = Cst * pow(temp,1.75)*pow((32.0+28.0)/(32.0*28.0),0.5)/
        (pressure * pow(pow(16.3, c13)+pow(18.0, c13),2.0));

    const double y_co  = 1.0/(co_ratio + 1.0);
    const double y_co2 = co_ratio/(co_ratio + 1.0);
    return 1.0/(y_co/D_CO + y_co2/D_CO2);
  }

  // -----------------------------------------------------------------------------
  double SurfaceO2_error( double& q,           double& co2CoRatio,
                          const double po2S,   const double gaspres,
                          const double tempg,  const double tempP,
                          const double po2Inf, const double c_s,
                          const double dp,     const double k1,
                          const double k2,     const double k_a_s,
                          const CharModel chmodel )
  {
    co2CoRatio = co2coratio_fun( po2S, tempP );
    const double of = (0.5 + co2CoRatio)/(1.0+ co2CoRatio);
    const double landa = -(1.0-of);
    const double do2mix = Do2_mix(gaspres,tempg,co2CoRatio);
    switch ( chmodel ){
      case FIRST_ORDER:
      case LH:
        q = q_fun(k1,k2,po2S);
        break;
      case FRACTAL:
        q = k_a_s *  po2S/ 8.314 / tempg; // bg : what temperature should be used here ?  tp, tg or (tp+tg)/2
        break;
      case CCK:
      case INVALID_CHARMODEL:
        assert(false);
        break;
    };

    return po2S/ gaspres - (landa + (po2Inf/ gaspres - landa)*exp(-q * dp/(2* c_s * do2mix)));
  }
  // -----------------------------------------------------------------------------
  double ka_fun( const double& theta, const double& S_g,
                 const double& rho,   const double& dp,
                 const double& temp )
  {
    const double chi_value = chi(theta, S_g, rho, dp);
    const double k = k_fun(temp);
    const double Q = Q_number(chi_value, k, temp);
    return 3*k*(1.0/(tanh(Q)*Q)-1.0/pow(Q,2.0));
  }
  // -----------------------------------------------------------------------------

} // namespace CHAR
