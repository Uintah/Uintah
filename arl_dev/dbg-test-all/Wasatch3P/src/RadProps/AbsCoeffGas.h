/**
 * \file   AbsCoeffGas.h
 * \author Lyubima Simeonova
 * \date   November, 2011
 *
 * Copyright (c) 2014 The University of Utah
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

#ifndef AbsCoeffGas_h
#define AbsCoeffGas_h

#include "RadiativeSpecies.h"
#include <string>
#include <vector>

/**
 *  \class SpeciesAbsCoeff
 *  \brief holds the spectrally resolved absorption coefficients at a single temperature
 */
class SpeciesAbsCoeff
{
public:

  SpeciesAbsCoeff ( const std::string& filename, const double temperature );
  ~SpeciesAbsCoeff(){};

  /**
   * @brief calculate the Planck mean absorption coefficient
   * @return the Planck-mean absorption coefficient at this temperature [1/cm]
   *
   * The Planck coefficient is determined via
   * \f[
   * \kappa_P = \frac{\int_0^\infty \kappa_\eta I_{b\eta} d\eta}{\int_0^\infty I_{b\eta}d\eta}
   *          = \frac{\pi}{\sigma T^4} \int_0^\infty \kappa_\eta I_{b\eta} d\eta
   * \f]
   * with
   * \f[
   * I_{b\eta} = 2hc^2\eta^3\left[\exp\left(\frac{h c\eta}{kT}\right)-1\right]^{-1}
   * \f]
   */
  double planck_abs_coeff() const;

  /**
   * @brief calculate the Rosseland absorption coefficient
   * @return the Rosseland-mean absorption coefficient at this temperature [1/cm]
   *
   * The Rosseland coefficient is determined via
   * \f[
   * \frac{1}{\kappa_R} = \frac{\int_0^\infty \frac{1}{\kappa_\eta}\frac{dI_{b\eta}}{dT}d\eta}{\int_0^\infty \frac{dI_{b\eta}}{dT}d\eta}
   *                    = \frac{\pi}{4\sigma T^3} \int_0^\infty \frac{1}{\kappa_\eta} \frac{dI_{b\eta}}{dT} d\eta
   * \f]
   * with
   * \f[
   * \frac{dI_{b\eta}}{dT} = \frac{c^3 h^2\eta^4}{kT^2} \frac{\exp\left(\frac{ch\eta}{kT}\right)}{\exp\left(\frac{ch\eta}{kT}\right)-1}
   * \f]
   */
  double rosseland_abs_coeff() const;

  /**
   * @brief calculate the effective absorption coefficient given an optical path length
   * @param opl [cm] the optical path length
   * @return the effective absorption coefficient, [1/cm]
   *
   * Determined via
   * \f[
   * \kappa_P = \frac{\int_0^\infty \kappa_\eta I_{b\eta} \exp(-\kappa_\eta L) d\eta}
   *                 {\int_0^\infty             I_{b\eta} \exp(-\kappa_\eta L) d\eta}
   * \f]
   * where L is the optical thickness.
   */
  double effective_abs_coeff( const double opl ) const;


  inline const std::vector<double>& get_coeffs() const{ return absCoeff_; }
  inline double min_wavenumber() const{ return loWaveNum_; }
  inline double max_wavenumber() const{ return hiWaveNum_; }
  inline double wavenumber_step() const{ return waveNumInc_; }
  inline size_t npts() const{ return npts_; }
  inline double temperature() const{ return temperature_; }

private:

  double temperature_;
  double waveNumInc_, loWaveNum_, hiWaveNum_;
  size_t npts_;
  std::vector<double> absCoeff_;
};


/**
 *  \struct SpeciesAbsData
 *  \brief holds the spectrally resolved absorption coefficients at a range of temperatures
 */
struct SpeciesAbsData{
  std::vector<SpeciesAbsCoeff> absCoeff;  ///< absorbtion coefficients at various temperatures
  std::vector<double> temperatures;       ///< temperatures at which we have absorbtion coefficients
  std::string speciesName;
};


/**
 *  \struct SpeciesGDataSingleTemperature
 */
struct SpeciesGDataSingleTemperature {
  std::vector<double> gvalues, kvalues;
  double temperature;
  void read( std::istream& file );
  void write( std::ostream& file ) const;
};


/**
 *  \struct SpeciesGData
 */
struct SpeciesGData {
  std::string speciesName;
  std::vector<SpeciesGDataSingleTemperature> data;
  void read( std::istream& file );
  void write( std::ostream& file ) const;
};


/**
 *  \class FSK
 */
class FSK
{
public:

# ifdef RadProps_ENABLE_PREPROCESSOR
  /**
   * @brief obtain and tabulate in a file the modified absorption coefficients "k" and the
   *    modified wavenumbers "g" for each radiative at a given set of temperatures for each species
   *    from spectral data
   *
   * The spectral data is contained in ".txt" files for a given species at a given temperatures
   * These files are obtained from the HTRAN and HITEMP databases.
   * Reference: "HITRAN on the web,"
   * The file names must have the following format
   * "AbsCoeff+SpeciesName+T+Temperature.txt", e.g. "AbsCoeffH2OT296.txt"
   *
   * @param mySpecies - order of radiative species
   * @param outputFileName - name of the file to write the FSK data to.
   * @param path - the directory to search for files.  Defaults to working directory
   */
  FSK( const std::vector<RadiativeSpecies>& mySpecies,
       const std::string outputFileName="Gs.txt",
       const std::string path="." );
# endif // RadProps_ENABLE_PREPROCESSOR

  ~FSK(){};

  /**
   * @brief Load the file containing the modified absorption coefficients "k" and the
   *    modified wavenumbers "g" for each radiative at a given set of temperatures
   * @param fileN - the file containing the k and g data for each radiative at a given set of tempertures
   */
  FSK( const std::string fileN );

  /**
   * @brief obtain the mixture absorption coefficient "k" at the given
   *        temperature and "g" value through linear interpolation
   * @param mixMoleFrac The mole fractions of the radiative species, ordered properly.
   * @param mixT mixture temperature
   * @param gp - user provided "g" value
   *
   * @return mixture absorption coefficient
   */
  double mixture_abs_coeff( const std::vector<double>& mixMoleFrac,
                            const double mixT,
                            const double gp ) const;

  /**
   *  \brief obtain the ordered list of species names that are considered here
   */
  const std::vector<RadiativeSpecies>& species() const{ return speciesOrder_; }

  /**
   *  Evaluate the function "a(wallT,medT,g)=dg_wall/dg_medium" which is
   *  the stretching factor between the k-distribution of the wall and the
   *  k-distribution of the medium. This function participates in the boundary
   *  condition at the wall of the modified g-dependent RTE.
   *
   *  Reference: Modest, Michael F., Radiative Heat transfer, 2nd edition, 2003, pp.619-620
   *
   * @param a a(wallT,medT,g)=dg_wall/dg_medium - a term in the wall b.c. of the modified g-dependent RTE
   * @param mixMoleFrac The mole fractions of the radiative species, ordered properly.
   * @param medT The temperature of the medium
   * @param wallT The temperature of the wall
   */
  void a_function( std::vector<double>& a,
                   const std::vector<double>& mixMoleFrac,
                   const double medT,
                   const double wallT ) const;

  /**
   * Evaluate the k-distribution of the mixture from the k-distribution of the
   * individual species at tabulated temperatures.
   *
   * The modified absorption coefficients:
   * \f[
   *   k(Tmix, gmix) = \sum_i x_i \sqrt(k(T_0,g_{T_0})k(T_1,g_{T_1})).
   * \f]
   * The modified wavenumber,
   * \f[
   *  gmix(Tmix) = \prod_i g_i(Tmix),
   *  \f]
   * where \f$ g_i(Tmix) \f$ is the modified wavenumber of radiative
   * species i, linearly interpolated from \f$ g_i(T_0) \f$ and \f$ g_i(T_1) \f$.
   *
   * Reference: Modest, Michael F., Radiative Heat transfer, 2nd edition, 2003, pp.630-635
   *
   * @param gmix the modified absorption coefficients, \f$ k(Tmix, gmix) = \sum_i x_i \sqrt(k(T_0,g_{T_0})k(T_1,g_{T_1})) \f$, in units of 1/cm.
   * @param kmix the modified wavenumber, \f$ gmix(Tmix) = \prod_i g_i(Tmix) \f$, in units of 1/cm.
   * @param mixMoleFrac
   * @param mixT mixture temperature, in units of Kelvin.
   */
  void mixture_coeffs( std::vector<double>& gmix,
                       std::vector<double>& kmix,
                       const std::vector<double>& mixMoleFrac,
                       const double mixT ) const;

private:

  const bool allowClipping_;
  std::vector<RadiativeSpecies> speciesOrder_;
  std::vector<SpeciesGData> spCalFSK_;

 };



/**
 * \class GreyGas
 *
 * \todo split mixture_coeffs into three methods for each of the coefficients.
 */
class GreyGas
{
  struct GreyGasData {
    std::string speciesName;
    int ntemp;
    std::vector<double> temperatures, planckCoeff, rossCoeff, effAbsCoeff;

    void read( std::ifstream& file );
    void write( std::ofstream& file ) const;
  };

public:

# ifdef RadProps_ENABLE_PREPROCESSOR
  /**
   * @brief obtain and tabulate in a file the grey gas Planck, Rosseland, and Effective absorption coefficients
   * for each radiative at a given set of temperatures from spectral data
   *
   * The spectral data is contained in ".txt" files for a given species at a given temperatures
   * These files are obtained from the HTRAN and HITEMP databases.
   * Reference: "HITRAN on the web,"
   * The file names must have the following format
   * "AbsCoeff+SpeciesName+T+Temperature.txt", e.g. "AbsCoeffH2OT296.txt"
   * @param MyspeciesGG - order of radiative species
   * @param opl - the optical path length (cm) for calculating the effective mean absorption coefficient
   * @param outputFileName - the name of the file to write the preprocessed results to.
   * @param path - the directory to look for files.  Defaults to working directory.
   */
  GreyGas( const std::vector<RadiativeSpecies>& mySpeciesGG,
           const double opl = 0.01,
           const std::string outputFileName="GreyGasProperties.txt",
           const std::string path="." );
# endif // RadProps_ENABLE_PREPROCESSOR

  ~GreyGas(){};

  /**
   * @brief Load the file containing the grey gas Planck, Rosseland, and Effective absorption coefficients
   * for each radiative at a given set of temperatures from spectral data
   * @param fileName - the preprocessed file containing the grey data coefficients for each radiative at a given set of temperatures
   *
   * @return
   */
  GreyGas( const std::string fileName );

  /**
   * @brief Calculate the grey gas Planck, Rosseland, and Effective
   *  absorption coefficients for the mixture from the pre-calculated
   *  and tabulated coefficients for the individual species.
   *  Linear interpolation in temperature.
   *
   * @param planckCff Mixture Planck absorption coefficient in units of 1/cm.
   * @param rossCff Mixture Rosseland absorption coefficient in units of 1/cm.
   * @param effCff Mixture Effective absorption coefficient in units of 1/cm.
   * @param mixMoleFrac mole fractions of the species participating in radiation.
   *        These are the species given by the species() method.
   * @param mixT mixture temperature in Kelvin
   */
  void mixture_coeffs( double& planckCff,
                       double& rossCff,
                       double& effCff,
                       const std::vector<double>& mixMoleFrac,
                       const double mixT ) const;

  /**
   *  \brief obtain the ordered list of species that are considered here
   */
  const std::vector<RadiativeSpecies>&
  species() const{ return speciesOrder_; }

  /**
   * @return the optical path length for the effective mean absorption coefficient
   */
  double optical_path_length() const{ return opl_; }

private:

  const bool allowClipping_;
  double loWaveNum_, hiWaveNum_, waveNumInc_; // wave nunber bounds and spacing
  size_t nspecies_;
  size_t npts_;
  std::vector<GreyGasData> data_;
  std::vector<RadiativeSpecies> speciesOrder_;
  double opl_;  ///< Optical path length
};

#endif // AbsCoeffGas_h
