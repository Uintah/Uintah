#ifndef SingleRateInformation_h
#define SingleRateInformation_h

/*
 *  
 *  \author Babak Goshayeshi (www.bgoshayeshi.com)
 *  \date   May, 2013 
 *                 
 *   Department of Chemical Engineering - University of Utah
 */
#include <CCA/Components/Wasatch/Coal/CoalData.h>

namespace SNGRATE {
	
  enum SingleRateSpecies {
    CO   = 0,
    H2   = 1,
    INVALID_SPECIES = 99
  };
  
	

  class SingleRateInformation{
  public:
    SingleRateInformation( const Coal::CoalComposition& coalType );
    double get_hydrogen_coefficient() const{ return h_;  }
    double get_oxygen_coefficient()   const{ return o_;  }
    double get_molecularweight()      const{ return mw_; }
    const double get_tarMonoMW()      const{return coalComp_.get_tarMonoMW(); };
  protected:
    double h_, o_, mw_;
    const Coal::CoalComposition& coalComp_;
  };

} // namespace end

#endif // SingleRateInformation_h
