
/*
 *  OutFluxVolume.h: 
 *
 *  Written by:
 *   John Schmidt
 *   Department of Mechanical Engineering
 *   University of Utah
 *   Nov 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_OutFluxVolume_H
#define UINTAH_EXCEPTIONS_OutFluxVolume_H

#include <Core/Exceptions/Exception.h>
#include <Core/Geometry/IntVector.h>
#include <string>

namespace Uintah {
  
  class OutFluxVolume : public SCIRun::Exception {
  public:
    OutFluxVolume(SCIRun::IntVector loc,double outflux, double vol);
    OutFluxVolume(const OutFluxVolume&);
    virtual ~OutFluxVolume();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string d_msg;
    OutFluxVolume& operator=(const OutFluxVolume&);
  };
  
} // End namespace Uintah

#endif


