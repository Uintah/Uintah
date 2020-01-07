#include "CharData.h"

#include <stdexcept>
#include <sstream>
#include <math.h>

namespace CHAR{

//--------------------------------------------------------------------

  Array2D binary_diff_coeffs( const double& press,
                              const double& temp )
  {
    /* This function assembles an array of binary diffusion coefficients using
     * a mysterious correlation found in the BYU version of the CCK code.
     * \param press: pressure (Pa)
     * \param temp : temperature (K)
     *
     * Order here is REALLY important!!!
     * [CO2 CO O2 H2 H2O CH4 N2]
     */
    const int n = 7;
    Array2D d( boost::extents[n][n] );

    // set diagonal values to zero
    for(Array2D::index j = 0; j != n; j++){
      d[j][j] = 0;
    }

    // temperature and pressure correlations
    const double ftp2 = pow(temp, 7.0/3.0)*press/101325.0; // H2O
    const double ftp1 = pow(temp, 5.0/3.0)*press/101325.0; // everything else

    //Now set non-diagonal values
    // CO2-X
    d[0][1] = 1.199e-09 * ftp1; d[1][0] = d[0][1]; // CO2-CO
    d[0][2] = 1.208e-09 * ftp1; d[2][0] = d[0][2]; // CO2-O2
    d[0][3] = 4.713e-09 * ftp1; d[3][0] = d[0][3]; // CO2-H2
    d[0][4] = 2.726e-11 * ftp2; d[4][0] = d[0][4]; // CO2-H2O
    d[0][5] = 1.365e-09 * ftp1; d[5][0] = d[0][5]; // CO2-CH4
    d[0][6] = 1.191e-09 * ftp1; d[6][0] = d[0][6]; // CO2-N2

    // CO-X
    d[1][2] = 1.537e-09 * ftp1; d[2][1] = d[1][2]; // CO-O2
    d[1][3] = 5.487e-09 * ftp1; d[3][1] = d[1][3]; // CO-H2
    d[1][4] = 4.272e-11 * ftp2; d[4][1] = d[1][4]; // CO-H2O
    d[1][5] = 1.669e-09 * ftp1; d[5][1] = d[1][5]; // CO-CH4
    d[1][6] = 1.500e-09 * ftp1; d[6][1] = d[1][6]; // CO-N2

    // O2-X
    d[2][3] = 5.783e-09 * ftp1; d[3][2] = d[2][3]; // O2-H2
    d[2][4] = 4.202e-11 * ftp2; d[4][2] = d[2][4]; // O2-H2O
    d[2][5] = 1.711e-09 * ftp1; d[5][2] = d[2][5]; // O2-CH4
    d[2][6] = 1.523e-09 * ftp1; d[6][2] = d[2][6]; // O2-N2

    // H2-X
    d[3][4] = 2.146e-10 * ftp2; d[4][3] = d[3][4]; // H2-H2O
    d[3][5] = 5.286e-09 * ftp1; d[5][3] = d[3][5]; // H2-CH4
    d[3][6] = 5.424e-09 * ftp1; d[6][3] = d[3][6]; // H2-N2

    // H2O-X
    d[4][5] = 4.059e-11 * ftp2; d[5][4] = d[4][5]; // H2O-CH4
    d[4][6] = 4.339e-11 * ftp2; d[6][4] = d[4][6]; // H2O-N2

    // CH4-X
    d[5][6] = 1.657e-09 * ftp1; d[6][5] = d[5][6]; // CH4-N2

    return d;
  }

//--------------------------------------------------------------------
  Vec effective_diff_coeffs( const Array2D& dBinary,
                             const Vec& x,
                             const double& epsilon,
                             const double& tau_f )
  {
    /*
     * This function calculates effective diffusivities for transport
     * within porous media (such as char) and puts them into a vector.
     * Formula is from [1].
     *
     * \param x    : vector of mole fractions
     * \param D       : array of binary diffusivities -- this includes N2
     * \param epsilon : particle (core) porosity
     * \param tau_f   : ratio of tortuosity and macroporosity
     */

    double xnp1  = 1.0 - std::accumulate(x.begin(), x.end(), 0.0);
    xnp1 = fmax(0.0, fmin(1.0, xnp1) );

    Vec effD; effD.clear();

    size_t n = x.size();
    for(size_t i = 0; i < n; i++){
      double sumTerm = xnp1/dBinary[i][n]; //
      for(size_t j = 0; j < n; j++){
        if(i != j){
          sumTerm += x[j]/dBinary[i][j];
        }
      }
      effD.push_back( epsilon*(1-x[i])/(sumTerm*tau_f) );
    }
    return effD;
  }

//--------------------------------------------------------------------

CharOxidationData::
CharOxidationData( const Coal::CoalType sel )
 : sel_    ( sel ),
  coalComp_( sel ),
  speciesData_( GasSpec::SpeciesData::self() )
{
  // I have call this class for two reason :
  // 1- Extract O  mass fraction from parent coal
  // 2- In case if I want to close the H balance over the coal particle I can use it.
  const Coal::CoalComposition coalComp( sel_ );
  o_           = coalComp.get_O();
  c_           = coalComp.get_C();
  fixedCarbon_ = coalComp.get_fixed_c();
  volatiles_   = coalComp.get_vm();
  set_data();
}

//--------------------------------------------------------------------
// =====>>>>>  I have to ask Prof Smith about T_ciritical of coal. !!!!!!    <<<<<<<=====
void 
CharOxidationData::
set_data()
{
  s0_    = 300E3; // m2/kg
  e0_    = 0.47;
  rPore_ = 6E-10; // Angstrom

  switch (sel_) {
    case Coal::NorthDakota_Lignite:
    case Coal::Gillette_Subbituminous:
    case Coal::MontanaRosebud_Subbituminous:
    case Coal::Illinois_Bituminous:
    case Coal::Kentucky_Bituminous:
    case Coal::Pittsburgh_Shaddix:
    case Coal::Black_Thunder:
    case Coal::Shenmu:
    case Coal::Guizhou:
    case Coal::Russian_Bituminous:
    case Coal::Utah_Bituminous:
    case Coal::Utah_Skyline:
    case Coal::Utah_Skyline_Char_10atm:
    case Coal::Utah_Skyline_Char_12_5atm:
    case Coal::Utah_Skyline_Char_15atm:
    case Coal::Highvale:
    case Coal::Highvale_Char:
    case Coal::Eastern_Bituminous:
    case Coal::Eastern_Bituminous_Char:
    case Coal::Illinois_No_6:
    case Coal::Illinois_No_6_Char:

      insufficient_data();
      break;

    case Coal::Pittsburgh_Bituminous:
      e0_ = 0.1;
      rPore_ = 6E-10; // Angstrom
      break;

    default:
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "Unsupported coal type" << std::endl
          << std::endl;
      throw std::runtime_error( msg.str() );
  }
}

//--------------------------------------------------------------------
// insufficient data for char oxidation calculation
void 
CharOxidationData::
insufficient_data()
{
  proc0cout << " ------------------------------------------------------------------------ \n"
            << " WARNING:\n"
            << "  For more accurate calculation in Char oxidation model, this coal type   \n"
            << "  requires the following values to be specified in CharData.cc:          \n"
            << "   - initial coal porosity                      (default = " << e0_ << ")\n"
            << "   - mean pore radius (m)                       (default = " << rPore_ << ")\n"
            << "   - initial internal char surface area (m2/kg) (default = " << s0_ << ")\n"
            << " ------------------------------------------------------------------------ \n";
}
//--------------------------------------------------------------------

// Initial Particle density, if the size the particle remains constant

  double initial_particle_density(double mass, double diameter){
    return mass/(pow(diameter, 3.0)*3.141592653589793/6);
  }

//--------------------------------------------------------------------

  const GasSpec::GasSpecies
  CharOxidationData::char_to_gas_species( CHAR::CharGasSpecies spec ) const
  {
    GasSpec::GasSpecies g2cSpec;
    switch(spec){
      case CHAR::O2  :  g2cSpec = GasSpec::O2 ;  break;
      case CHAR::CO2 :  g2cSpec = GasSpec::CO2;  break;
      case CHAR::CO  :  g2cSpec = GasSpec::CO ;  break;
      case CHAR::H2  :  g2cSpec = GasSpec::H2 ;  break;
      case CHAR::H2O :  g2cSpec = GasSpec::H2O;  break;
      case CHAR::CH4 :  g2cSpec = GasSpec::CH4;  break;

      default:
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "Invalid char species." << std::endl
            << std::endl;
        throw std::runtime_error( msg.str() );
    }
    return g2cSpec;
  }

//--------------------------------------------------------------------

  const double
  CharOxidationData::get_mw( CHAR::CharGasSpecies spec ) const
  {
    return speciesData_.get_mw( char_to_gas_species(spec) );
  }

//--------------------------------------------------------------------


} // namespace CHAR

/*
[1] Lewis et. al. Pulverized Steam Gasification Rates of Three Bituminous Coal Chars in an Entrained-Flow
    Reactor at Pressurized Conditions. Energy and Fuels. 2015, 29, 1479-1493.
    http://pubs.acs.org/doi/abs/10.1021/ef502608y
*/
