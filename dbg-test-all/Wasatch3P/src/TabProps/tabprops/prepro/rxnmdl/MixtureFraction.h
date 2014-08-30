/*
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

#ifndef MIXTURE_FRACTION
#define MIXTURE_FRACTION

#include <vector>

/** test the mixture fraction class.  Returns true for pass, false for fail */
bool perform_mixfrac_tests();


/*
 *  the Constituents class must provide the following:
 *     elemental composition of a species
 *     species and elemental molecular weights
 *     species and elemental names
 */
namespace Cantera{ class Constituents; }   // forward declaration


//--------------------------------------------------------------------
/**
 *  @class MixtureFraction
 *  @brief Provides tools related to the mixture fraction.
 *
 *  @author James C. Sutherland
 *  @date   February, 2005
 *
 *  The MixtureFraction class is intended for use in two-stream mixing
 *  problems.  It requires information about species molecular weights
 *  and elemental composition.  Currently this information must be provided
 *  by Cantera.  However, this could easily be abstracted into a general
 *  object of which Cantera could be a particular case...
 *
 *  The default implementation uses Bilger's mixture fraction.  This
 *  is easily changed, however, as the set_gammas() method is virtual.
 *  Deriving from this class and overloading set_gammas() will facilitate
 *  other definitions of the mixture fraction.
 *
 *  See
 *      Sutherland, Smith, & Chen
 *      "Quantification of differential diffusion in nonpremixed systems"
 *      Combustion Theory & Modelling
 *      May, 2005 Volume 9, Number 2, p. 365
 */
class MixtureFraction
{
 public:

  /** Constructor
   *  @brief Construct a mixture fraction object
   *
   *  @param specProps : Cantera object which provides information such as
   *                     molecular and elemental weights, species atomic
   *                     composition, etc.
   *  @param oxidFrac  : vector of oxidizer mass or mole fractions
   *  @param fuelFrac  : vector of   fuel   mass or mole fractions
   *  @param inputMassFrac : flag set to true if MASS fractions are provided.
   */
  MixtureFraction( Cantera::Constituents & specProps,
		   const std::vector<double> & oxidFrac,
		   const std::vector<double> & fuelFrac,
		   const bool inputMassFrac );
  /** Constructor
   *  @brief Construct a skeletal mixture fraction object.
   *
   *  The MixtureFraction object cannot be used until fuel and oxidizer
   *  compositions are specified.  Then the initialize() method must be
   *  called to complete initialization.  If the MixtureFraction object
   *  is provided stream compositions at construction, the initialize()
   *  method need not be called.
   */
  MixtureFraction( Cantera::Constituents & specProps );

  /** destructor */
  virtual ~MixtureFraction();

  /**
   * @brief Set the mass fractions in the fuel stream
   *  @param fuelMassFrac : Vector of mass fractions in the fuel stream.
   */
  void set_fuel_mass_frac( std::vector<double> & fuelMassFrac )
    { fuelMassFrac_ = fuelMassFrac; };

  /**
   *  @brief Set the mass fractions in the oxidizer stream
   *  @param oxidMassFrac : Vector of mass fractions in the oxidizer stream.
   */
  void set_oxid_mass_frac( std::vector<double> & oxidMassFrac )
    { oxidMassFrac_ = oxidMassFrac; };

  /**
   *  @brief Initialize everything.
   *  @param oxid : Oxidizer composition vector
   *  @param fuel : Fuel composition vector
   *  @param massFrac : true if composition is in mass fractions,
   *                    false for mole fraction.
   *  If the short constructor is used, then the user must set the
   *  oxidizer and fuel compositions and then call the initialize()
   *  method to complete initialization.
   */
  void initialize( const std::vector<double> & oxid,
		   const std::vector<double> & fuel,
		   const bool massFrac );

  /** @brief Check to see if the mixture fraction object is ready for use. */
  const bool is_ready() const
    { return ready_; };

  /** @brief Return the stoichiometric mixture fraction */
  inline double stoich_mixfrac() const
    { return stoichMixfrac; };

  /** Get fuel and oxidizer composition as double pointers */
  inline const double* fuel_massfr() const {return &(fuelMassFrac_[0]);};
  inline const double* oxid_massfr() const {return &(oxidMassFrac_[0]);};

  /** Get fuel and oxidizer composition as std::vector<double> */
  inline const std::vector<double> & fuel_massfr_vec() const {return fuelMassFrac_; };
  inline const std::vector<double> & oxid_massfr_vec() const {return oxidMassFrac_; };

  /**
   *  @brief Convert species composition to mixture fraction
   *
   *  @param massFrac : Vector of species MASS fractions
   *  @param mixFrac  : The resulting mixture fraction
   */
  void species_to_mixfrac( const std::vector<double> & massFrac,
			   double & mixFrac );

  /**
   *  @brief Convert mixture fraction to unreacted species mass fractions
   *
   *  @param mixFrac  : The mixture fraction
   *  @param massFrac : The UNREACTED mixture mass fractions obtained
   *                    by pure mixing of the fuel and oxidizer streams
   */
  void mixfrac_to_species( const double mixFrac,
			   std::vector<double> & massFrac ) const;

  /**
   *  @brief Compute equivalence ratio from mixture fraction
   *  @param mixFrac : The mixture fraction
   */
  double mixfrac_to_equiv_ratio( const double mixFrac ) const;

  /**
   *  @brief Compute mixture fraction from equivalence ratio
   *  @param eqRat : The equivalence ratio
   */
  double equiv_ratio_to_mixfrac( const double eqRat ) const;

  /**
   *  @brief Estimate the products of COMPLETE combustion
   *  @param mixFrac  : The mixture fraction.
   *  @param massFrac : Product composition for COMPLETE combustion
   *  @param calcMassFrac : set true to return mass fractions,
   *                        false to return mole fractions.
   *
   *  The Burke-Schumann approximation for nonpremixed combustion
   *  assumes complete and infinitely fast combustion with products
   *  of CO2 and H2O.  This method calculates the products of
   *  complete combustion.  If the composition is rich (i.e. mixture
   *  fraction greater than stoichiometric) then the product
   *  composition includes unburnt fuel, while if the composition
   *  is lean, the resulting product composition includes oxidizer.
   *  Species compositions are thus piecewise linear functions of the
   *  mixture fraction.
   */
  void estimate_product_comp( const double mixFrac,
  			      std::vector<double> & massFrac,
			      const bool calcMassFrac );

  //------------------------------------------------------------------

 protected:

  virtual void set_gammas();

  /** @brief Calculate the stoichiometric mixture fraction and return its value. */
  double compute_stoich_mixfrac() const;

  double compute_beta( const std::vector<double> & massFrac );

  void compute_elem_mass_frac( const std::vector<double> & spec,
			       std::vector<double> & elem ) const;

  void set_stoichiometry();

  Cantera::Constituents & specProps_;

  const int nelem_, nspec_;

  double stoichMixfrac;   // stoichiometric mixture fraction
  double beta0_, beta1_;  // coupling functions in fuel (1) and oxidizer (0)

  bool ready_;

  std::vector<double> gamma_;       // (nelem)       elemental weighting factors
  std::vector<double> elemMassFr_;  // work space

  std::vector<double> fuelMassFrac_, oxidMassFrac_;

  std::vector<double> specMolWt_, elemWt_;

  // normalized stoichiometric coefficients of products and reactants,
  // used for estimate_product_comp()
  std::vector<double> stoichProdMassFrac_;

  //------------------------------------------------------------------
 private:

  MixtureFraction( const MixtureFraction & );             // no copying
  MixtureFraction & operator = (const MixtureFraction& ); // no assignment

};

#endif
