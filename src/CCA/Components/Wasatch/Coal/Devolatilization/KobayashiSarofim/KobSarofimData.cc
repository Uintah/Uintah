/*
 *  KobSarrofimData.cc
 *
 *  Created by Babak Goshayeshi on 6/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "KobSarofimData.h"

namespace SAROFIM {

  KobSarofimInformation::
  KobSarofimInformation( const Coal::CoalComposition& coalType )
  : coalcomp_ ( coalType )
  {
    const double c = coalcomp_.get_C();
    const double h = coalcomp_.get_H();
    const double o = coalcomp_.get_O();

    const double nc = c/12.0;
    const double nh = h/1.0;
    const double no = o/16.0;

    h_ = nh/nc;
    o_ = no/nc;

    mw_= 12.0 + h_ + o_ * 16.0;
  }

} // namespace end
