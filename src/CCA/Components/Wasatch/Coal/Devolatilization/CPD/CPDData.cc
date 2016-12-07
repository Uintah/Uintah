#include "CPDData.h"

#include <stdexcept>
#include <sstream>
#include <numeric>
#include <iostream>

using std::endl;
using std::ostringstream;
using std::vector;

namespace CPD{


  //------------------------------------------------------------------

  CPDInformation::CPDInformation( const Coal::CoalType sel )
    : nspec_   ( 15    ),
      coalType_( sel   ),
      coalComp_( sel   ),
      l0_      ( 0.094 ),  // wt fraction
      Ml0_     ( 28    ),
      coordNo_ ( 3.6   ),  // coordination number of coal lattice         [3]
      lbPop0_  (0.59   ),  // initial normalized labile bridge population [3]
      speciesData_()
 
  {
    /*  For more information on the data here:
     *
     *  [1]  Serio, M. A., Hamblen, D. G., Markham, J. R., & Solomon, P. R.,
     *       Energy & Fuels, 1(2), 138-152 (1987).
     *
     *  [2] Solomon, P., & Hamblen, D.
     *      Energy & Fuels, 2(4), 405-422 (1988).
     *   
     *  [3] Grant, D. M., Pugmire, R. J., Fletcher, T. H., & Kerstein, A. R.,
     *      Energy & Fuels,  3, 175-186 (1989).    
     */

    Mw_ = 41.12; // For NorthDakota_Lignite - A function must be defined for that
    switch (sel) {

    /* jtm- Note:
     * fg_ given below is a vector of functional group *mass*
     * fractions. If I am correct, they need to be converted to mole
     * fractions for all calculations.
     */

    case Coal::NorthDakota_Lignite:
      fg_.push_back( 0.065 ); // Y1  CO2 extra loose
      fg_.push_back( 0.030 ); // Y2  CO2 loose
      fg_.push_back( 0.005 ); // Y3  tight
      fg_.push_back( 0.061 ); // Y4  H2o loose
      fg_.push_back( 0.033 ); // Y5  tight
      fg_.push_back( 0.060 ); // Y6  CO ether loose
      fg_.push_back( 0.044 ); // Y7  CO ether tight
      fg_.push_back( 0.006 ); // Y8  HCN loose
      fg_.push_back( 0.012 ); // Y9  HCN tight
      fg_.push_back( 0.001 ); // Y10 NH3
      fg_.push_back( 0.000 ); // Y12 methane extra loose
      fg_.push_back( 0.016 ); // Y13 methane loose
      fg_.push_back( 0.009 ); // Y14 methane tight
      fg_.push_back( 0.017 ); // Y15 H aromatic
      fg_.push_back( 0.090 ); // Y17 CO extra tight

      tarMassFrac0_ = 0.095;
      Mw_           = 41.12;
      coordNo_      = 3.5;  // [3]
      lbPop0_       = 0.61; // [3]
      break;
			
    case Coal::Gillette_Subbituminous:
      fg_.push_back( 0.018 );
      fg_.push_back( 0.053 );
      fg_.push_back( 0.028 );
      fg_.push_back( 0.031 );
      fg_.push_back( 0.031 );
      fg_.push_back( 0.080 );
      fg_.push_back( 0.043 );
      fg_.push_back( 0.007 );
      fg_.push_back( 0.015 );
      fg_.push_back( 0.000 );
      fg_.push_back( 0.000 );
      fg_.push_back( 0.026 );
      fg_.push_back( 0.017 );
      fg_.push_back( 0.012 );
      fg_.push_back( 0.031 );

      tarMassFrac0_ = 0.158;
      break;
            
    case Coal::MontanaRosebud_Subbituminous:
    case Coal::Highvale:
      fg_.push_back( 0.035 ); // Y1  CO2 extra loose
      fg_.push_back( 0.035 ); // Y2  CO2 loose
      fg_.push_back( 0.030 ); // Y3  tight
      fg_.push_back( 0.051 ); // Y4  H2o loose
      fg_.push_back( 0.051 ); // Y5  tight
      fg_.push_back( 0.055 ); // Y6  CO ether loose
      fg_.push_back( 0.013 ); // Y7  CO ether tight
      fg_.push_back( 0.005 ); // Y8  HCN loose
      fg_.push_back( 0.015 ); // Y9  HCN tight
      fg_.push_back( 0.001 ); // Y10 NH3
      fg_.push_back( 0.000 ); // Y12 methane extra loose
      fg_.push_back( 0.022 ); // Y13 methane loose
      fg_.push_back( 0.012 ); // Y14 methane tight
      fg_.push_back( 0.013 ); // Y15 H aromatic
      fg_.push_back( 0.000 ); // Y17 CO extra tight

      tarMassFrac0_ = 0.127;
      coordNo_ = 5.8;  // [3]
      lbPop0_  = 0.56; // [3]
      break;

    case Coal::Illinois_Bituminous:
      fg_.push_back( 0.022 ); // Y1  CO2 extra loose
      fg_.push_back( 0.022 ); // Y2  CO2 loose
      fg_.push_back( 0.030 ); // Y3  tight
      fg_.push_back( 0.045 ); // Y4  H2o loose
      fg_.push_back( 0.000 ); // Y5  tight
      fg_.push_back( 0.060 ); // Y6  CO ether loose
      fg_.push_back( 0.063 ); // Y7  CO ether tight
      fg_.push_back( 0.010 ); // Y8  HCN loose
      fg_.push_back( 0.016 ); // Y9  HCN tight
      fg_.push_back( 0.000 ); // Y10 NH3
      fg_.push_back( 0.011 ); // Y12 methane extra loose
      fg_.push_back( 0.011 ); // Y13 methane loose
      fg_.push_back( 0.022 ); // Y14 methane tight
      fg_.push_back( 0.016 ); // Y15 H aromatic
      fg_.push_back( 0.000 ); // Y17 CO extra tight

      tarMassFrac0_ = 0.081;
      coordNo_ = 5.2; // [3]
      break;

    case Coal::Kentucky_Bituminous:
      fg_.push_back( 0.000 ); // Y1  CO2 extra loose
      fg_.push_back( 0.006 ); // Y2  CO2 loose
      fg_.push_back( 0.005 ); // Y3  tight
      fg_.push_back( 0.011 ); // Y4  H2o loose
      fg_.push_back( 0.011 ); // Y5  tight
      fg_.push_back( 0.050 ); // Y6  CO ether loose
      fg_.push_back( 0.026 ); // Y7  CO ether tight
      fg_.push_back( 0.026 ); // Y8  HCN loose
      fg_.push_back( 0.009 ); // Y9  HCN tight
      fg_.push_back( 0.000 ); // Y10 NH3
      fg_.push_back( 0.020 ); // Y12 methane extra loose
      fg_.push_back( 0.015 ); // Y13 methane loose
      fg_.push_back( 0.015 ); // Y14 methane tight
      fg_.push_back( 0.012 ); // Y15 H aromatic
      fg_.push_back( 0.020 ); // Y17 CO extra tight

      tarMassFrac0_ = 0.183;
      break;

    case Coal::Pittsburgh_Bituminous:
      fg_.push_back( 0.000 ); // Y1  CO2 extra loose
      fg_.push_back( 0.006 ); // Y2  CO2 loose
      fg_.push_back( 0.005 ); // Y3  tight
      fg_.push_back( 0.011 ); // Y4  H2o loose
      fg_.push_back( 0.011 ); // Y5  tight
      fg_.push_back( 0.050 ); // Y6  CO ether loose
      fg_.push_back( 0.022 ); // Y7  CO ether tight
      fg_.push_back( 0.009 ); // Y8  HCN loose
      fg_.push_back( 0.022 ); // Y9  HCN tight
      fg_.push_back( 0.000 ); // Y10 NH3
      fg_.push_back( 0.020 ); // Y12 methane extra loose
      fg_.push_back( 0.015 ); // Y13 methane loose
      fg_.push_back( 0.015 ); // Y14 methane tight
      fg_.push_back( 0.012 ); // Y15 H aromatic
      fg_.push_back( 0.020 ); // Y17 CO extra tight

      tarMassFrac0_ = 0.190;
      Mw_ = 45.96;    // [3]
      coordNo_ = 5.0; // [3]
      break;            
     
    case Coal::Pittsburgh_Shaddix:
        fg_.push_back( 0.000 ); // Y1  CO2 extra loose
        fg_.push_back( 0.006 ); // Y2  CO2 loose
        fg_.push_back( 0.005 ); // Y3  tight
        fg_.push_back( 0.011 ); // Y4  H2o loose
        fg_.push_back( 0.011 ); // Y5  tight
        fg_.push_back( 0.050 ); // Y6  CO ether loose
        fg_.push_back( 0.022 ); // Y7  CO ether tight
        fg_.push_back( 0.009 ); // Y8  HCN loose
        fg_.push_back( 0.022 ); // Y9  HCN tight
        fg_.push_back( 0.000 ); // Y10 NH3
        fg_.push_back( 0.020 ); // Y12 methane extra loose
        fg_.push_back( 0.015 ); // Y13 methane loose
        fg_.push_back( 0.015 ); // Y14 methane tight
        fg_.push_back( 0.012 ); // Y15 H aromatic
        fg_.push_back( 0.020 ); // Y17 CO extra tight

        tarMassFrac0_ = 0.190;
        Mw_ = 45.96;        
        break;
        
    case Coal::Russian_Bituminous: // Interpolation
        fg_.push_back( 0.0096 );
        fg_.push_back( 0.0135 );
        fg_.push_back( 0.0138 );
        fg_.push_back( 0.0238 );
        fg_.push_back( 0.0145 );
        fg_.push_back( 0.0528 );
        fg_.push_back( 0.0304 );
        fg_.push_back( 0.0142 );
        fg_.push_back( 0.0155 );
        fg_.push_back( 0.0001 );
        fg_.push_back( 0.0153 );
        fg_.push_back( 0.0152 );
        fg_.push_back( 0.0160 );
        fg_.push_back( 0.0130 );
        fg_.push_back( 0.0130 );

        tarMassFrac0_ = 0.1562;
        Mw_ = 47.3348;
        break;
        
    case Coal::Black_Thunder: // Interpolation
        
        fg_.push_back( 0.0371 );
        fg_.push_back( 0.0353 );
        fg_.push_back( 0.0219 );
        fg_.push_back( 0.0478 );
        fg_.push_back( 0.0299 );
        fg_.push_back( 0.0638 );
        fg_.push_back( 0.0403 );
        fg_.push_back( 0.0068 );
        fg_.push_back( 0.0143 );
        fg_.push_back( 0.0005 );
        fg_.push_back( 0.0023 );
        fg_.push_back( 0.0189 );
        fg_.push_back( 0.0144 );
        fg_.push_back( 0.0146 );
        fg_.push_back( 0.0350 );

        tarMassFrac0_ = 0.1155;
        Mw_ = 41.0397;
        break;
        
    case Coal::Shenmu: //Interpolation
        
        fg_.push_back( 0.0108 );
        fg_.push_back( 0.0145 );
        fg_.push_back( 0.0147 );
        fg_.push_back( 0.0252 );
        fg_.push_back( 0.0156 );
        fg_.push_back( 0.0530 );
        fg_.push_back( 0.0305 );
        fg_.push_back( 0.0137 );
        fg_.push_back( 0.0155 );
        fg_.push_back( 0.0002 );
        fg_.push_back( 0.0146 );
        fg_.push_back( 0.0154 );
        fg_.push_back( 0.0160 );
        fg_.push_back( 0.0130 );
        fg_.push_back( 0.0122 );

        tarMassFrac0_ = 0.1532;
        Mw_ = 43.5368;              
        break;
    
      case Coal::Guizhou: //Interpolation
      case Coal::Eastern_Bituminous:
        
        fg_.push_back( 0.0098 );
        fg_.push_back( 0.0137 );
        fg_.push_back( 0.0139 );
        fg_.push_back( 0.0240 );
        fg_.push_back( 0.0148 );
        fg_.push_back( 0.0528 );
        fg_.push_back( 0.0303 );
        fg_.push_back( 0.0140 );
        fg_.push_back( 0.0155 );
        fg_.push_back( 0.0002 );
        fg_.push_back( 0.0151 );
        fg_.push_back( 0.0152 );
        fg_.push_back( 0.0160 );
        fg_.push_back( 0.0130 );
        fg_.push_back( 0.0129 );

        tarMassFrac0_ = 0.1559;
        Mw_ = 47.3576;
        break;
				
      case Coal::Utah_Bituminous: //Interpolation
        
        fg_.push_back( 0.0140 );
        fg_.push_back( 0.0169 );
        fg_.push_back( 0.0176 );
        fg_.push_back( 0.0294 );
        fg_.push_back( 0.0168 );
        fg_.push_back( 0.0539 );
        fg_.push_back( 0.0325 );
        fg_.push_back( 0.0126 );
        fg_.push_back( 0.0155 );
        fg_.push_back( 0.0002 );
        fg_.push_back( 0.0130 );
        fg_.push_back( 0.0154 );
        fg_.push_back( 0.0163 );
        fg_.push_back( 0.0133 );
        fg_.push_back( 0.0099 );

        tarMassFrac0_ = 0.1437;
        Mw_ = 47.1; // ??? Check again 
        break;
        

    default:
      ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << endl
          << "No 'fg' data available for coal type: '" << coal_type_name(sel) << "'" << endl
          << endl;
      throw std::runtime_error( msg.str() );
    }

    mwVec_.push_back( speciesData_.get_mw(GasSpec::CO2) ); //1  CO2  -
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CO2) ); //2  CO2
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CO2) ); //3  CO2
    mwVec_.push_back( speciesData_.get_mw(GasSpec::H2O) ); //4  H2O  -
    mwVec_.push_back( speciesData_.get_mw(GasSpec::H2O) ); //5  H2O
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CO ) ); //6  CO   -
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CO ) ); //7  CO
    mwVec_.push_back( speciesData_.get_mw(GasSpec::HCN) ); //8  HCN  -
    mwVec_.push_back( speciesData_.get_mw(GasSpec::HCN) ); //9  HCN
    mwVec_.push_back( speciesData_.get_mw(GasSpec::NH3) ); //10 NH3
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CH4) ); //11 CH4  -
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CH4) ); //12 CH4
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CH4) ); //13 CH4
    mwVec_.push_back( speciesData_.get_mw(GasSpec::H  ) ); //14 H    -
    mwVec_.push_back( speciesData_.get_mw(GasSpec::CO ) ); //15 CO   -

    sumfg_ = std::accumulate( fg_.begin(), fg_.end(), 0.0 ) + tarMassFrac0_;

    A0_.push_back( 0.81E13 ); // CO2 extra loose ( 1)
    A0_.push_back( 0.65E17 ); // CO2 loose       ( 2)
    A0_.push_back( 0.11E16 ); // CO2 tight       ( 3)
    A0_.push_back( 0.22E19 ); // H2O loose       ( 4)
    A0_.push_back( 0.17E14 ); // H2O tight       ( 5)
    A0_.push_back( 0.14E19 ); // CO ehter loose  ( 6)
    A0_.push_back( 0.15E16 ); // CO ehter tight  ( 7)
    A0_.push_back( 0.17E14 ); // HCN loose       ( 8)
    A0_.push_back( 0.69E13 ); // HCN tight       ( 9)
    A0_.push_back( 0.12E13 ); // NH3             (10)
    A0_.push_back( 0.84E15 ); // CH4 extra loose (12)
    A0_.push_back( 0.75E14 ); // CH4 loose       (13)
    A0_.push_back( 0.34E12 ); // CH4 tight       (14)
    A0_.push_back( 0.10E15 ); // H aromatic      (15)
    A0_.push_back( 0.20E14 ); // CO extra tight  (17)
		
    // These are E0/R
    E0_.push_back( 22500 );
    E0_.push_back( 33850 );
    E0_.push_back( 38315 );
    E0_.push_back( 30000 );
    E0_.push_back( 32700 );
    E0_.push_back( 40000 ); // 6
    E0_.push_back( 40500 );
    E0_.push_back( 30000 );
    E0_.push_back( 42500 );
    E0_.push_back( 27300 );
    E0_.push_back( 30000 );
    E0_.push_back( 30000 );
    E0_.push_back( 30000 );
    E0_.push_back( 40500 );
    E0_.push_back( 45500 );	
	
    // These are ???/R
    sigma_.push_back( 1500 );
    sigma_.push_back( 1500 );
    sigma_.push_back( 2000 );
    sigma_.push_back( 1500 );
    sigma_.push_back( 1500 );
    sigma_.push_back( 6000 );
    sigma_.push_back( 1500 );
    sigma_.push_back( 1500 );
    sigma_.push_back( 4750 );
    sigma_.push_back( 3000 );
    sigma_.push_back( 1500 );
    sigma_.push_back( 2000 );
    sigma_.push_back( 2000 );
    sigma_.push_back( 6000 );
    sigma_.push_back( 1500 );

  }

  //------------------------------------------------------------------

  double CPDInformation::get_l0_mole() const
  {
    return l0_*Mw_/Ml0_;
  }


  //------------------------------------------------------------------

  SpeciesSum::SpeciesSum()
  {
    // note: ordering here is important
    sContVec_.push_back( species_connect( CO2  ) );
    sContVec_.push_back( species_connect( H2O  ) );
    sContVec_.push_back( species_connect( CO   ) );
    sContVec_.push_back( species_connect( HCN  ) );
    sContVec_.push_back( species_connect( NH3  ) );
    sContVec_.push_back( species_connect( CH4  ) );
    sContVec_.push_back( species_connect( H    ) );
  }

  //------------------------------------------------------------------

  const SpeciesSum&
  SpeciesSum::self()
  {
    static SpeciesSum s;
    return s;
  }

  //------------------------------------------------------------------

  SpeciesSum::~SpeciesSum(){}

  //------------------------------------------------------------------

  MWCompPair
  SpeciesSum::species_connect( const CPDSpecies spec )
  {
    double mw;
    VecI sList;

    switch ( spec ) {

    case CO2:
      sList.push_back(1);
      sList.push_back(2);
      sList.push_back(3);
      mw = 44.0;    
      break;

    case H2O:
      sList.push_back(4);
      sList.push_back(5);
      mw = 18.0;    
      break;

    case CO:
      sList.push_back(6);
      sList.push_back(7);
      sList.push_back(15);
      mw = 28.0;    
      break;

    case HCN:
      sList.push_back(8);
      sList.push_back(9);
      mw = 27.0;    
      break;

    case NH3:
      sList.push_back(10);
      mw = 17.0;    
      break;

    case CH4:
      sList.push_back(11);
      sList.push_back(12);
      sList.push_back(13);
      mw = 16.0;    
      break;

    case H:
      sList.push_back(14);
      mw = 1.0;    
      break;

    default:
      ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << endl
          << "Unsupported species choice" << endl
          << endl;
      throw std::runtime_error( msg.str() );
    }
    return MWCompPair( mw, sList );
  }


  //------------------------------------------------------------------
  vector<double>
  deltai_0( const CPDInformation& info, const double c0 )
  {
    /* This function returns a vector whose elements are the side chain
     * mass fractions within the volatile mass, excluding labile and char
     * bridges.
     */
    vector<double> result1;
    const vector<double>& fg = info.get_fgi();
    const int nspec = info.get_nspec();
    const double sumfg = info.get_sumfg();
    const double l0 = info.get_l0_mass(); // ?????? mole or mass ?
	
    for (int i=0; i<nspec; ++i) {
      result1.push_back( (1.0 - c0 - l0 ) * fg[i] / sumfg );
    }
    return result1;
  }

  //------------------------------------------------------------------
  // calculates mass fraction of tar in volatile mass
  double
  tar_0( const CPDInformation& info, const double c0 )
  {
    const double l0       = info.get_l0_mass();
    const double tarFrac0 = info.get_tarMassFrac();
    const double sumfg    = info.get_sumfg();
    return (1.0 - c0 - l0 ) * tarFrac0 / sumfg;
  }

  //------------------------------------------------------------------

} // namespace CPD
