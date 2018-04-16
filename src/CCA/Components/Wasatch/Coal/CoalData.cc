#include <stdexcept>
#include <sstream>

#include "CoalData.h"

using std::string;
using std::ostringstream;
using std::endl;

namespace Coal{

  //------------------------------------------------------------------

  string coal_type_name( const CoalType ct )
  {
    string name;
    switch (ct) {
      case NorthDakota_Lignite:          name = "NorthDakota_Lignite";          break;
      case Gillette_Subbituminous:       name = "Gillette_Subbituminous";       break;
      case MontanaRosebud_Subbituminous: name = "MontanaRosebud_Subbituminous"; break;
      case Illinois_Bituminous:          name = "Illinois_Bituminous";          break;
      case Kentucky_Bituminous:          name = "Kentucky_Bituminous";          break;
      case Pittsburgh_Bituminous:        name = "Pittsburgh_Bituminous";        break;
      case Pittsburgh_Shaddix:           name = "Pittsburgh_Shaddix";           break;
      case Russian_Bituminous:           name = "Russian_Bituminous";           break;
      case Black_Thunder:                name = "Black_Thunder";                break;
      case Shenmu:                       name = "Shenmu";                       break;
      case Guizhou:                      name = "Guizhou";                      break;
      case Utah_Bituminous:              name = "Utah_Bituminous";              break;
      case Utah_Skyline:                 name = "Utah_Skyline";                 break;
      case Utah_Skyline_Char_10atm:      name = "Utah_Skyline_Char_10atm";      break;
      case Utah_Skyline_Char_12_5atm:    name = "Utah_Skyline_Char_12_5atm";    break;
      case Utah_Skyline_Char_15atm:      name = "Utah_Skyline_Char_15atm";      break;
      case Highvale:                     name = "Highvale";                     break;
      case Highvale_Char:                name = "Highvale_Char";                break;
      case Eastern_Bituminous:           name = "Eastern_Bituminous";           break;
      case Eastern_Bituminous_Char:      name = "Eastern_Bituminous_Char";      break;
      case Illinois_No_6:                name = "Illinois_No_6";                break;
      case Illinois_No_6_Char:           name = "Illinois_No_6_Char";           break;


      default:
        ostringstream msg;
        msg << endl
            << __FILE__ << " : " << __LINE__ << endl
            << "Unsupported coal type" << endl
            << endl;
        throw std::runtime_error( msg.str() );
    }
    return name;
  }

  CoalType coal_type( const std::string& coaltype )
  {
    if     ( coaltype == coal_type_name(Coal::NorthDakota_Lignite         )) return Coal::NorthDakota_Lignite;
    else if( coaltype == coal_type_name(Coal::Gillette_Subbituminous      )) return Coal::Gillette_Subbituminous;
    else if( coaltype == coal_type_name(Coal::MontanaRosebud_Subbituminous)) return Coal::MontanaRosebud_Subbituminous;
    else if( coaltype == coal_type_name(Coal::Illinois_Bituminous         )) return Coal::Illinois_Bituminous;
    else if( coaltype == coal_type_name(Coal::Kentucky_Bituminous         )) return Coal::Kentucky_Bituminous;
    else if( coaltype == coal_type_name(Coal::Pittsburgh_Bituminous       )) return Coal::Pittsburgh_Bituminous;
    else if( coaltype == coal_type_name(Coal::Pittsburgh_Shaddix          )) return Coal::Pittsburgh_Shaddix;
    else if( coaltype == coal_type_name(Coal::Russian_Bituminous          )) return Coal::Russian_Bituminous;
    else if( coaltype == coal_type_name(Coal::Black_Thunder               )) return Coal::Black_Thunder;
    else if( coaltype == coal_type_name(Coal::Shenmu                      )) return Coal::Shenmu;
    else if( coaltype == coal_type_name(Coal::Guizhou                     )) return Coal::Guizhou;
    else if( coaltype == coal_type_name(Coal::Utah_Bituminous             )) return Coal::Utah_Bituminous;
    else if( coaltype == coal_type_name(Coal::Utah_Skyline                )) return Coal::Utah_Skyline;
    else if( coaltype == coal_type_name(Coal::Utah_Skyline_Char_10atm     )) return Coal::Utah_Skyline_Char_10atm;
    else if( coaltype == coal_type_name(Coal::Utah_Skyline_Char_12_5atm   )) return Coal::Utah_Skyline_Char_12_5atm;
    else if( coaltype == coal_type_name(Coal::Utah_Skyline_Char_15atm     )) return Coal::Utah_Skyline_Char_15atm;
    else if( coaltype == coal_type_name(Coal::Highvale                    )) return Coal::Highvale;
    else if( coaltype == coal_type_name(Coal::Highvale_Char               )) return Coal::Highvale_Char;
    else if( coaltype == coal_type_name(Coal::Eastern_Bituminous          )) return Coal::Eastern_Bituminous;
    else if( coaltype == coal_type_name(Coal::Eastern_Bituminous_Char     )) return Coal::Eastern_Bituminous_Char;
    else if( coaltype == coal_type_name(Coal::Illinois_No_6               )) return Coal::Illinois_No_6;
    else if( coaltype == coal_type_name(Coal::Illinois_No_6_Char          )) return Coal::Illinois_No_6_Char;
    else{
      ostringstream msg;
      msg << endl
          << __FILE__ << " : " << __LINE__ << endl
          << "Unsupported coal type: '" << coaltype << "'" << endl
          << endl;
      throw std::runtime_error( msg.str() );
    }
    return INVALID_COALTYPE;
  }

  //------------------------------------------------------------------

  string species_name( const GasSpeciesName spec )
  {
    string name="";
    switch( spec ){
      case O2 : name="O2";  break;
      case CO : name="CO";  break;
      case CO2: name="CO2"; break;
      case H2 : name="H2";  break;
      case H2O: name="H2O"; break;
      case CH4: name="CH4"; break;
      case HCN: name="HCN"; break;
      case NH3: name="NH3"; break;
      case H  : name="H";   break;
      case INVALID_SPECIES: name=""; break;
    }
    return name;
  }

  //------------------------------------------------------------------

  GasSpeciesName gas_name_to_enum( const std::string name )
  {
    if     ( name == "O2"  ) return O2;
    else if( name == "CO"  ) return CO ;
    else if( name == "CO2" ) return CO2;
    else if( name == "H2"  ) return H2 ;
    else if( name == "H2O" ) return H2O;
    else if( name == "CH4" ) return CH4;
    else if( name == "HCN" ) return HCN;
    else if( name == "NH3" ) return NH3;
    else if( name == "H"   ) return H  ;
    return INVALID_SPECIES;
  }

  //------------------------------------------------------------------

  CoalComposition::CoalComposition( const Coal::CoalType sel )
  {
    /*  For more information on the data here:
     *
     *  Serio, M. A., Hamblen, D. G., Markham, J. R., & Solomon, P. R.,
     *  Energy & Fuels, 1(2), 138-152 (1987).
     *
     *  Solomon, P., & Hamblen, D.
     *  Energy & Fuels, 2(4), 405-422 (1988).
     */

    tarMonoMW_ = 128.17;

    switch (sel) {
      case NorthDakota_Lignite:
        // Mole Fraction ( daf )0.4588    0.3946    0.0065    0.0028    0.1372
        // Mass Fraction (daf) 0.665 0.048 0.011 0.011 0.265
        C_ = 0.665;
        H_ = 0.048;
        N_ = 0.011;
        S_ = 0.011;
        O_ = 0.265;
        moisture_ = 0.299; // wt fraction
        ash_ = 0.072; // wt fraction
        vm_ = 0.295;
        fixedc_ = 0.334;
        break;

      case Gillette_Subbituminous:
        // Mole Fraction DAF 0.4950    0.3851    0.0071    0.0013    0.1115
        // Mass Fraction (daf)  0.720 0.047 0.012 0.005 0.216
        C_ = 0.720;
        H_ = 0.047;
        N_ = 0.012;
        S_ = 0.005;
        O_ = 0.216;
        moisture_ = 0.27930;  // Check again with other references
        ash_ = 0.05570;// Check again with other references
        break;

      case MontanaRosebud_Subbituminous:
        // Mole Fraction DAF  0.4908    0.3959    0.0070    0.0030    0.1033
        // Mass Fraction (daf) 0.724 0.049 0.012 0.012 0.203      C = 0.4908;
        C_ = 0.724;
        H_ = 0.049;
        N_ = 0.012;
        S_ = 0.012;
        O_ = 0.203;
        moisture_ = 0.213;
        ash_ = 0.118;
        break;

      case Illinois_Bituminous:
        // Mass Fraction (daf) 0.736 0.047 0.014 0.038 0.165
        C_ = 0.736;
        H_ = 0.047;
        N_ = 0.014;
        S_ = 0.038;
        O_ = 0.165;
        moisture_ = 0.101;
        ash_ = 0.073;
        vm_ = 0.359;
        fixedc_ = 0.467;
        break;

      case Kentucky_Bituminous:
        // Mass Fraction (daf)  0.817 0.056 0.019 0.024 0.084
        C_ = 0.817;
        H_ = 0.056;
        N_ = 0.019;
        S_ = 0.024;
        O_ = 0.084;
        moisture_ = 0.086;
        ash_ = 0.233;
        vm_ = 0.352;
        fixedc_ = 0.415;
        break;

      case Pittsburgh_Bituminous:
        // Mass Fraction (daf) 0.821 0.056 0.017 0.024 0.082
        C_ = 0.821;
        H_ = 0.056;
        N_ = 0.017;
        S_ = 0.024;
        O_ = 0.082;
        moisture_ = 0.01;
        ash_ = 0.069;
        vm_ = 0.289;
        fixedc_ = 0.632;
        break;

      case Pittsburgh_Shaddix:
        // Mass Fraction (daf) 0.821 0.056 0.017 0.024 0.082
        C_ = 0.829;
        H_ = 0.056;
        N_ = 0.016;
        S_ = 0.022;
        O_ = 0.077;
        moisture_ = 0.014;
        ash_ = 0.069;
        vm_ = 0.354;
        fixedc_ = 0.563;
        break;

      case Russian_Bituminous: // [1]
        C_ = 0.6604;
        H_ = 0.0418;
        N_ = 0.0199;
        S_ = 0.0206;
        O_ = 0.0707;
        moisture_ = 0.031;
        ash_      = 0.149;
        vm_       = 0.30;
        fixedc_   = 0.52;
        break;

      case Black_Thunder: // [2]
        C_ = 0.641;
        H_ = 0.055;
        N_ = 0.009;
        S_ = 0.005;
        O_ = 0.291;
        moisture_ = 0.108;
        ash_      = 0.050;
        vm_       = 0.404;
        fixedc_   = 0.438;
        break;

      case Shenmu: // [2]
        C_ = 0.868;
        H_ = 0.052;
        N_ = 0.014;
        S_ = 0.010;
        O_ = 0.057;
        moisture_ = 0.057;
        ash_      = 0.087;
        vm_       = 0.351;
        fixedc_   = 0.505;
        break;

      case Guizhou: // [2]
        C_ = 0.840;
        H_ = 0.053;
        N_ = 0.016;
        S_ = 0.015;
        O_ = 0.076;
        moisture_ = 0.057;
        ash_      = 0.318;
        vm_       = 0.228;
        fixedc_   = 0.397;
        break;

      case Utah_Bituminous:
        C_ = 0.710;
        H_ = 0.060;
        N_ = 0.013;
        S_ = 0.005;
        O_ = 0.127;
        moisture_ = 0.024;
        ash_      = 0.083;
        vm_       = 0.456;
        fixedc_   = 0.437;
        break;

      case Utah_Skyline:
        C_ = 0.710;
        H_ = 0.060;
        N_ = 0.013;
        S_ = 0.005;
        O_ = 0.127;
        moisture_ = 0.0241;
        ash_      = 0.0787;
        vm_       = 0.4706;
        fixedc_   = 0.4266; //calculated by difference

        break;

      case Utah_Skyline_Char_10atm:
        // composition of the parent coal
        C_ = 0.710;
        H_ = 0.060;
        N_ = 0.013;
        S_ = 0.005;
        O_ = 0.127;
        // These data are for char generated from Utah Skyline coal at 10atm.
        // See Tables 3 & 5 from [3] for more details.
        moisture_ = 0.0;
        ash_      = 0.278;
        vm_       = 0.0;
        fixedc_   = 0.722;
        break;

      case Utah_Skyline_Char_12_5atm:
        // composition of the parent coal
        C_ = 0.710;
        H_ = 0.060;
        N_ = 0.013;
        S_ = 0.005;
        O_ = 0.127;
        // These data are for char generated from Utah Skyline coal at 10atm.
        // See Tables 3 & 5 from [3] for more details.
        moisture_ = 0.0;
        ash_      = 0.248;
        vm_       = 0.0;
        fixedc_   = 0.752;
        break;

      case Utah_Skyline_Char_15atm:
        // composition of the parent coal
        C_ = 0.710;
        H_ = 0.060;
        N_ = 0.013;
        S_ = 0.005;
        O_ = 0.127;
        // These data are for char generated from Utah Skyline coal at 10atm.
        // See Tables 3 & 5 from [3] for more details.
        moisture_ = 0.0;
        ash_      = 0.236;
        vm_       = 0.0;
        fixedc_   = 0.764;
        break;

      case Highvale:
        C_ = 0.6923;
        H_ = 0.0457;
        N_ = 0.0095;
        S_ = 0.0032;
        O_ = 0.2493;

        moisture_ = 0.01;
        ash_      = 0.1215;
        vm_       = 0.3978;
        fixedc_   = 0.4707;
        break;

      case Highvale_Char:
        // composition of the parent coal
        C_ = 0.6923;
        H_ = 0.0457;
        N_ = 0.0095;
        S_ = 0.0032;
        O_ = 0.2493;

        moisture_ = 0.0;
        ash_      = 0.206;
        vm_       = 0.0;
        fixedc_   = 0.794;
        break;

      case Eastern_Bituminous:
        C_ = 0.8487;
        H_ = 0.0557;
        N_ = 0.0159;
        S_ = 0.0105;
        O_ = 0.0692;

        moisture_ = 0.0075;
        ash_      = 0.0902;
        vm_       = 0.3491;
        fixedc_   = 0.5532;
        break;

      case Eastern_Bituminous_Char:
        // composition of the parent coal
        C_ = 0.8487;
        H_ = 0.0557;
        N_ = 0.0159;
        S_ = 0.0105;
        O_ = 0.0692;

        moisture_ = 0.0;
        ash_      = 0.1402;
        vm_       = 0.0;
        fixedc_   = 0.8598;
        break;

      case Illinois_No_6:
        // composition of the parent coal
        C_ = 0.7851;
        H_ = 0.0549;
        N_ = 0.0136;
        S_ = 0.0483;
        O_ = 0.0981;

        moisture_ = 0.0964;
        ash_      = 0.0799;
        vm_       = 0.3678;
        fixedc_   = 0.4558;
        break;

      case Illinois_No_6_Char:
        // composition of the parent coal
        C_ = 0.7851;
        H_ = 0.0549;
        N_ = 0.0136;
        S_ = 0.0483;
        O_ = 0.0981;

        moisture_ = 0.0;
        ash_      = 0.1492;
        vm_       = 0.0;
        fixedc_   = 0.8508;
        break;

      default:
        C_ = H_ = N_ = S_ = O_ = moisture_ = ash_ = vm_ = fixedc_ = 0.0;
        ostringstream msg;
        msg << endl
            << __FILE__ << " : " << __LINE__ << endl
            << "Coal composition unavailable for '" << coal_type_name(sel) << "'" << endl
            << endl;
        throw std::runtime_error( msg.str() );
    }
  }

  bool
  CoalComposition::test_data(){
    bool check_;
    check_ = true;
    if ((C_+H_+N_+S_+O_ != 1.0) or (fixedc_+vm_+ash_+moisture_ != 1.0)) {
      check_ = false;
    }
    return check_;
  }

  //------------------------------------------------------------------

  double CoalComposition::get_c0() const
  {
    double c0=0;
    if( get_C() > 0.859 ) c0 = 11.83 * get_C() - 10.16;
    if( get_O() > 0.125 ) c0 = 1.4 * get_O() - 0.175;
    return c0;
  }
} // namespace Coal

/*
 [1] Jovanovic, Rastko, Aleksandra Milewska, Bartosz Swiatkowski, Adrian Goanta, and Hartmut Spliethoff
     Numerical investigation of influence of homogeneous/heterogeneous ignition/combustion mechanisms on
     ignition point position during pulverized coal combustion in oxygen enriched and recycled flue
     gases atmosphere. International Journal of Heat and Mass Transfer 54, no. 4 (January 31, 2011): 921-931.
     http://linkinghub.elsevier.com/retrieve/pii/S0017931010005740.

 [2] Liu, Yinhe, Manfred Geier, Alejandro Molina, and Christopher R. Shaddix.Pulverized coal stream ignition
     delay under conventional and oxy-fuel combustion conditions. International Journal of Greenhouse
     Gas Control (June 2011).
     http://linkinghub.elsevier.com/retrieve/pii/S1750583611000880.


 [3] Lewis et. al. Pulverized Steam Gasification Rates of Three Bituminous Coal Chars in an Entrained-Flow
     Reactor at Pressurized Conditions. Energy and Fuels. 2015, 29, 1479-1493.
     http://pubs.acs.org/doi/abs/10.1021/ef502608y

 */

