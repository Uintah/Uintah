#ifndef SARROFIM_KobSarofimInformation_h
#define SARROFIM_KobSarofimInformation_h

/*
 *  KobSarofimData.h
 *  ODT
 *
 *  Created by Babak Goshayeshi on 6/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <CCA/Components/Wasatch/Coal/CoalData.h>

namespace SAROFIM {
	
	enum KobSarofimSpecies {
    CO   = 0,
    H2   = 1,
    INVALID_SPECIES = 99
  };
  
	

  class KobSarofimInformation{
  public:
    KobSarofimInformation( const Coal::CoalComposition& coalType );
    const double get_hydrogen_coefficient() const{return h_; };
    const double get_oxygen_coefficient()   const{return o_; };
    const double get_molecularweight()      const{return mw_;};
    const double get_tarMonoMW()            const{return coalcomp_.get_tarMonoMW(); };
  protected:
    double h_, o_, mw_;
    const Coal::CoalComposition& coalcomp_;
  };

} // namespace end

#endif // SARROFIM_KobSarofimInformation_h
