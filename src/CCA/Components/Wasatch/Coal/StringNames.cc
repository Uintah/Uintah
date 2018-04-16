#include "StringNames.h"

using std::string;

namespace Coal {

const StringNames&
StringNames::self()
{
  static StringNames s;
  return s;
}
//--------------------------------------------------------------------
StringNames::StringNames()
: dev_char("dev_char_production"), cpd_l("cpd_l"),
  cpd_g("cpd_g_"), cpd_delta("cpd_delta_"),
  cpd_kg("cpd_kg_"), cpd_y("cpd_y_"), cpd_dy("cpd_dy_"), dev_mv("Volatile_Mass"),
  cpd_kb("cpd_kb"), cpd_C_rhs("cpd_C_rhs"), dev_volatile_src("dev_volatile_rhs"),
  cpd_product_rhs ("cpd_product_rhs"), cpd_charProd_rhs("cpd_char_production_"),
  char_mass("char_mass"),
  char_coco2ratio("char_Mole_CO2/CO"),
  char_heattogas("heat_released_to_gas"),
  char_co2_rhs("char_CO2_rhs"),
  char_co_rhs("char_CO_rhs"),
  char_o2_rhs("char_O2_rhs"),
  char_h2_rhs("char_H2_rhs"),
  char_h2o_rhs("char_H2O_rhs"),
  char_ch4_rhs("char_CH4_rhs"),
  char_oxid_rhs("char_oxidation_rhs"),
  char_gasifh2o("H2O_Gasification_rate"),
  char_gasifco2("CO2_Gasification_rate"),
  char_gasifh2("H2_Gasification_rate"),
  ash_density("ash_density"), ash_porosity("ash_porosity"), ash_thickness("ash_film_thickness"),
  core_diameter("p_core_diameter"), core_density("p_core_density"), ash_mass_frac("ash_mass_fraction"),
  char_mass_frac("char_mass_fraction"), char_conversion("char_conversion"),
  log_freq_dist("log_frequency_distribution"), therm_anneal("thermal_annealing_factor"),
  coal_density ("Particle_Density"), coal_prtmass("Particle_Mass"),
  coal_mash("Ash_content"), coal_temprhs("coal_Temperature_rhs"),

  char_mass_rhs("char_mass_rhs"),
  cpd_tar("cpd_tar"), cpd_lbPopulation("cpd_lbPopulation"),
  sarofim_tar("sarofim_tar"), sarofim_tar_src("sarofim_tar_src"),
  singlerate_tar("singlerate_tar"), singlerate_tar_src("singlerate_tar_src"),
  moisture("moisture_mass")
{}
//--------------------------------------------------------------------
StringNames::~StringNames()
{}
//--------------------------------------------------------------------
} // coal namespace
