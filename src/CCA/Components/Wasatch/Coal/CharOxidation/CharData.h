#ifndef CharData_h
#define CharData_h

#include <cmath>
#include <boost/multi_array.hpp>
#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/CPDData.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>
#include <CCA/Components/Wasatch/Coal/SpeciesData.h>
#include "CharBase.h"

namespace CHAR{

  typedef std::vector<double>           Vec;
  typedef boost::multi_array<double, 2> Array2D;

  /* Calculates a matrix of binary diffusion coefficients using the correlation
   * used in the BYU version of the CCK code. At the moment, its origin is
   * unknown
   */
  Array2D binary_diff_coeffs( const double& press,
                             const double& temp );
  /* Calculates a vector of effiective diffusivities. Porosity (epsilon) and/or the
   * ratio of tortuosity to fraction of total porosity in the macrorpores (tau_f)
   * may be passed to calculate intraparticle effective diffusivites.
   *
   * The output has the following order:
   * [CO2 CO O2 H2 H2O CH4 N2]
   */
  Vec effective_diff_coeffs( const Array2D& dBinary,
                             const Vec&    x,
                             const double& epsilon = 1.0,
                             const double& tau_f   = 1.0 );

  /*
   *  Calculating initial particle density based on its initial mass mass recent size
   */
  double initial_particle_density(double mass, double diameter);

  /**
   *  \ingroup CharOxidation
   *  \class CharOxidationData
   *
   *  
   *  "Study on coal chars combustion under O2/CO2 atmosphere with
   *   fractal random pore model", Hua Fei, Song Hu, Jun Xiang, Lushi
   *   Sun, Peng Fu, Gang Chen, Fuel 90 (2011) 441–448.
   *
   */
  class CharOxidationData {
  public:
    /**
     *  \param coalType the specification of the type of coal to
     *         consider for the char oxidation model.
     */
    CharOxidationData( const Coal::CoalType coalType=Coal::Pittsburgh_Bituminous );
    
    
    const Coal::CoalComposition& get_coal_composition() const{ return coalComp_; } ///< return the CoalComposition
   
    /**
     *  \brief returns \f$\varepsilon_{0}\f$ - initial porosity of coal particle
     */
    double get_e0() const {return e0_;};

    /**
     *  \brief returns \f$r_{pore}\f$ - mean pore radius (m)
     */
    double get_r_pore() const {return rPore_;};

    /**
     *  \brief returns \f$S_{0}\f$ - internal surface area of initial char (m2/kg)
     */
    double get_S_0() const {return s0_;};

   /**
     *  \brief returns Oxygen mass fraction of coal
     */
    double get_O() const{ return o_;};

    /**
      *  \brief returns Carbon mass fraction of coal
      */
     double get_C() const{ return c_;};

     /**
      * \brief Returns fixed carbon mass fraction of coal
      */
     double get_fixed_C() const{ return fixedCarbon_;}

     /**
      * \brief Returns volatile mass fraction of coal
      */
     double get_vm() const{ return volatiles_;}

     /**
       *  \brief returns selected coal type
       */
     Coal::CoalType get_coal_type() const{ return sel_;};

     /**
       *  \brief returns a GasSpec::GasSpecies given a CHAR::CharGasSpecies
       */
     const GasSpec::GasSpecies char_to_gas_species( CHAR::CharGasSpecies spec ) const;

     /**
       *  \brief returns molecular weight of a given species
       */
     const double get_mw( CHAR::CharGasSpecies spec ) const;

    /**
     * Returns number of species involved in the char oxidation mechanism
     * Right now the value is hard coded...similar to CPD data gas phase species data
     * remember we might change the number of species in the future....
     */
    int get_ncomp() const {return 3 ;};
   
  protected:
    double e0_, rPore_, s0_, o_, c_, fixedCarbon_, volatiles_;
    Coal::CoalType sel_;
    Coal::CoalComposition coalComp_;
    void insufficient_data();
    void set_data();

  private:
    GasSpec::SpeciesData speciesData_;
  };

} // namespace CHAR

#endif  // OxidationFunctions_h
