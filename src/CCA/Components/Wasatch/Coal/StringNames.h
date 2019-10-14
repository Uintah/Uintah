#ifndef StringNames_COAL_h
#define StringNames_COAL_h

#include <string>

/**
 *   Author : Babak Goshayeshi
 *   Date   : Jan 2011
 *   University of Utah - Institute for Clean and Secure Energy
 *
 *   Holds names for Tags.
 */
namespace Coal {
  
class StringNames
{
public:
  static const StringNames& self();
  const std::string dev_char, cpd_l;
  const std::string cpd_g, cpd_delta, cpd_kg, cpd_y, cpd_dy, dev_mv;
  const std::string cpd_kb, cpd_C_rhs, dev_volatile_src, cpd_product_rhs, cpd_charProd_rhs;
  const std::string char_mass, char_coco2ratio, char_heattogas;
  const std::string char_co2_rhs, char_co_rhs, char_o2_rhs, char_h2_rhs, char_h2o_rhs, char_ch4_rhs;
  const std::string char_oxid_rhs, char_gasifh2o, char_gasifco2, char_gasifh2;
  const std::string ash_density, ash_porosity, ash_thickness, core_diameter, core_density;
  const std::string ash_mass_frac, char_mass_frac, char_conversion, log_freq_dist, therm_anneal;
  const std::string coal_density, coal_prtmass, coal_mash, coal_temprhs, char_mass_rhs;
  const std::string cpd_tar, cpd_lbPopulation;
  const std::string sarofim_tar, sarofim_tar_src;
  const std::string singlerate_tar, singlerate_tar_src;
  const std::string moisture;
  const std::string coalConsumedGasYi;
  const std::string coalProducedGasYi;
  const std::string coalYiConsumptionRate;
  const std::string coalYiProductionRate;
  const std::string coalTotalGasConsumptionRate;
  const std::string coalTotalGasProductionRate;
  const std::string coalConsumedGasEnthalpy;
  const std::string coalProducedGasEnthalpy;
  const std::string coalConsumedGasEnthalpySrc;
  const std::string coalProducedGasEnthalpySrc;
  const std::string heatFromCharRxns;
private:
  StringNames();    ~StringNames();
  StringNames(const StringNames&); 
  StringNames& operator=(const StringNames&); 
};

} // namespace Coal
#endif
