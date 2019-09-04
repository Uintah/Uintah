/*
 *  SingleRateData.cc
 *
 *  \author Babak Goshayeshi (www.bgoshayeshi.com)
 *  \date   May, 2013 
 *                 
 *   Department of Chemical Engineering - University of Utah
 */

#include <CCA/Components/Wasatch/Coal/Devolatilization/SingleRate/SingleRateData.h>

namespace SNGRATE {

		// Copied from KobSarofimInformation !
  SingleRateInformation::
  SingleRateInformation( const Coal::CoalComposition& coalType )
  : coalComp_ ( coalType )
  {
    const double c = coalComp_.get_C();
    const double h = coalComp_.get_H();
    const double o = coalComp_.get_O();

    const double nc = c/12.0;
    const double nh = h/1.0;
    const double no = o/16.0;

    h_ = nh/nc;
    o_ = no/nc;

    mw_= 12.0 + h_ + o_ * 16.0;
  }

} // namespace SNGRATE
