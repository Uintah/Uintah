/**
 *  \file   RadiativeSpecies.h
 *  \date   January, 2012
 *  \author Lyubima Simeonova
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

#ifndef Particles_h
#define Particles_h

#include <complex>
#include <vector>

class LagrangeInterpolant2D;  // forward declaration


/**
 * \fn double soot_abs_coeff( const double lambda, const double volFrac, const std::complex<double> refIndex );
 *
 * @return the absorption coefficient for soot given by
 * \f[ \kappa_\lambda = \frac{36\pi n k f_v}{\lambda((n^2-k^2+2)^2+4n^2k^2)} \f]
 * Units: 1/m
 *
 * @param lambda the wavelength (m)
 * @param volFrac volume fraction of soot
 * @param refIndex the index of refraction, \f$ m=n-ik \f$
 */
double soot_abs_coeff( const double lambda,
                       const double volFrac,
                       const std::complex<double> refIndex );

/**
 *  \class ParticleRadCoeffs
 *  \brief holds the spectrally resolved and effective absorbtion efficiencies multiplied by \f$\pi r^{2}\f$ for particles
 */
class ParticleRadCoeffs
{
public:

  /**
   * @param refIndex the refractive index of the particle.
   */
  ParticleRadCoeffs( const std::complex<double> refIndex );

  ~ParticleRadCoeffs();

  /**
   * @return \f$\frac{\kappa_\lambda}{N_T}\f$ - the spectral absorption
   *  coefficient divided by the particle number density, Units: (m<sup>2</sup>)/particle
   *
   * @param wavelength (m)
   * @param r  particle radius (m)
   *
   * The particle spectral absorption coefficient is given as
   * \f[\kappa_\lambda = \pi r^2 N_T Q_{abs}\f]
   * This method returns
   * \f$\frac{\kappa_\lambda}{N_T}\f$
   * You should multiply by the particle number density, \f$N_T\f$ to obtain
   * the actual particle absorption coefficient.
   */
  double abs_spectral_coeff( const double wavelength,
                             const double r ) const;


  /**
   * @return \f$\frac{\sigma_{s\lambda}}{N_T}\f$ - the scattering coefficient
   *  divided by the particle number density, in units of m<sup>2</sup>/particle.
   *
   * @param wavelength (m)
   * @param r  radius (m)
   *
   * The spectral scattering coefficient is given by
   * \f[ \sigma_{s\lambda} = \pi r^2 N_T Q_{sca}\f]
   * where \f$Q_{sca}\f$ is the scattering efficiency and \f$N_T\f$ is the
   * particle number density.
   */
  double scattering_spectral_coeff( const double wavelength,
                                    const double r ) const;
  /**
   * @return \f$\frac{\kappa_{P}}{N_T}\f$ - the Planck-mean absorption
   *  coefficient divided by the particle number density, (m<sup>2</sup>)/particle.
   *
   * @param r  radius (m)
   * @param T  temperature (K)
   *
   * The particle Planck-mean absorption coefficient is given as
   * \f[\kappa_P = \pi r^2 N_T Q_{abs,P}\f]
   * This method returns
   * \f$\frac{\kappa_P}{N_T}\f$
   * You should multiply by the particle number density, \f$N_T\f$ to obtain
   * the actual particle absorption coefficient.
   */
  double planck_abs_coeff( const double r,
                           const double T ) const;

  /**
   * @return \f$\frac{\sigma_{P}}{N_T}\f$ - the Planck-mean scattering
   *  coefficient divided by the particle number density, (m<sup>2</sup>)/particle.
   *
   * @param r  radius (m)
   * @param T  temperature (K)
   *
   * The particle Planck-mean scattering coefficient is given as
   * \f[\sigma_P = \pi r^2 N_T Q_{sca,P}\f]
   * This method returns
   * \f$\frac{\sigma_P}{N_T}\f$
   * You should multiply by the particle number density, \f$N_T\f$ to obtain
   * the actual particle scattering coefficient.
   */
  double planck_sca_coeff( const double r,
                           const double T) const;

  /**
   * @return \f$\frac{\kappa_{R}}{N_T}\f$ - the Rosseland-mean absorption
   *  coefficient divided by the particle number density, (m<sup>2</sup>)/particle.
   *
   * @param r  radius (m)
   * @param T  temperature (K)
   *
   * The particle Rosseland-mean absorption coefficient is given as
   * \f[\kappa_R = \pi r^2 N_T Q_{abs,R}\f]
   * This method returns
   * \f$\frac{\kappa_P}{N_T}\f$
   * You should multiply by the particle number density, \f$N_T\f$ to obtain
   * the actual particle absorption coefficient.
   */
  double ross_abs_coeff( const double r,
                         const double T ) const;

  /**
   * @return \f$\frac{\sigma_{R}}{N_T}\f$ - the Rosseland-mean scattering
   *  coefficient divided by the particle number density, (m<sup>2</sup>)/particle.
   *
   * @param r  radius (m)
   * @param T  temperature (K)
   *
   * The particle Rosseland-mean scattering coefficient is given as
   * \f[\sigma_R = \pi r^2 N_T Q_{sca,R}\f]
   * This method returns
   * \f$\frac{\sigma_R}{N_T}\f$
   * You should multiply by the particle number density, \f$N_T\f$ to obtain
   * the actual particle scattering coefficient.
   */
  double ross_sca_coeff( const double r,
                         const double T ) const;


private:


  void spectral_effs( std::vector<double>& absSpectralEff,
                      std::vector<double>& scaSpectralEff,
                      std::vector<double>& lambdas,
                      std::vector<double>& raV,
                      const double ramax,
                      const double ramin,
                      const std::complex<double> refIndex );

  /**
   * @brief Calculates the spectral absorption and scattering efficiencies multiplied by \f$\pi r^{2}\f$ using Mie theory
   * and tabulates them for a vector of radii and wavelengths
   * \f$\kappa_\lambda = \pi r^2 Q_{abs}\f$. Units: 1/m
   * \f$\sigma_{s\lambda} = \pi r^2 Q_{sca}\f$. Units: 1/m
   * @param absSpectralCoeff - vector of spectral absorption efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param scaSpectralCoeff - vector of spectral scattering efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param absSpectralEff - vector of spectral absorption efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param scaSpectralEff - vector of spectral scattering efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param lambdas - vector of wavelengths (m)
   * @param raV - vector of radii (m)
   */
  void spectral_coeff (std::vector<double>& absSpectralCoeff,
                       std::vector<double>& scaSpectralCoeff,
                       const std::vector<double>& absSpectralEff,
                       const std::vector<double>& scaSpectralEff,
                       const std::vector<double>& lambdas,
                       const std::vector<double>& raV );

  /**
   * @brief Calculates the the Planck-mean and Rosseland-mean absorption and scattering efficiencies multiplied by \f$\pi r^{2}\f$ 
   * and tabulates them for a vector of radii and temperatures
   * \f$\kappa_P = \pi r^2 Q_{abs P}\f$. Units: 1/m
   * \f$\sigma_{s P} = \pi r^2 Q_{sca P}\f$. Units: 1/m
   * \f$\kappa_R = \pi r^2 Q_{abs R}\f$. Units: 1/m
   * \f$\sigma_{s R} = \pi r^2 Q_{sca R}\f$. Units: 1/m
   * \f$\bar{Q_{P}}=\frac{15}{\pi^{4}}\int_{0}^{\infty}Q_{\xi}\xi^{3}(e^{\xi}-1)^{-1}\: d\xi\f$
   * \f$\frac{1}{\bar{Q_{R}}}=\frac{15}{\pi^{4}}\int_{0}^{\infty}\frac{1}{Q_{\xi}}\xi^{4}e^{\xi}(e^{\xi}-1)^{-2}\: d\xi\f$
   * where \f$\xi=C_{2}/{\lambda T}\f$ and \f$C_{2}=0.0144\f$ mK is Planck's second radiation constant,
   * and \f$Q_{\xi}\f$ is the respective efficiency calculated at the appropriate wavelength and radius
   * @param absPlanckCoeff - vector of Planck-mean absorption efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param scaPlanckCoeff - vector of Planck-mean scattering efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param absRossCoeff - vector of Rosseland-mean absorption efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param scaRossCoeff - vector of Rosseland-mean scattering efficiencies multiplied by \f$\pi r^{2}\f$. Units: (m<sup>2</sup>)/particle
   * @param absSpectralEff - vector of spectral absorption efficiencies Q_abs
   * @param scaSpectralEff - vector of spectral scattering efficiencies Q_sca
   * @param lambdas - vector of wavelengths (m)
   * @param raV - vector of radii (m)
   * @param tempV - vector of temperatures (K)
   * @param minT - minimum temperature to be tabulated (K)
   * @param maxT - maximum temperature to be tabulated (K)
   */

  void mean_coeff( std::vector<double>& absPlanckCoeff,
                   std::vector<double>& scaPlanckCoeff,
                   std::vector<double>& absRossCoeff,
                   std::vector<double>& scaRossCoeff,
                   const std::vector<double>& absSpectralEff,
                   const std::vector<double>& scaSpectralEff,
                   const std::vector<double>& lambdas,
                   const std::vector<double>& raV,
                   std::vector<double>& tempV,
                   const double minT,
                   const double maxT );

  LagrangeInterpolant2D* interp2AbsSpectralCoeff_;
  LagrangeInterpolant2D* interp2ScaSpectralCoeff_;

  LagrangeInterpolant2D* interp2AbsPlanckCoeff_;
  LagrangeInterpolant2D* interp2ScaPlanckCoeff_;
  LagrangeInterpolant2D* interp2AbsRossCoeff_;
  LagrangeInterpolant2D* interp2ScaRossCoeff_;

};
#endif // Particles_h

