
/*
 *  MaxIteration.h: 
 *
 *  Written by:
 *   John Schmidt
 *   Department of Mechanical Engineering
 *   University of Utah
 *   Nov 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_MaxIteration_H
#define UINTAH_EXCEPTIONS_MaxIteration_H

#include <Core/Exceptions/Exception.h>
#include <Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using namespace std;
  class MaxIteration : public SCIRun::Exception {
  public:
    MaxIteration(SCIRun::IntVector c,
                const int count, 
                const int n_passes,
                const int L_indx, 
                const string mes,
                const char* file,
                int line);
    
    MaxIteration(const MaxIteration&);
    virtual ~MaxIteration();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string d_msg;
    MaxIteration& operator=(const MaxIteration&);
  };
  
} // End namespace Uintah

#endif


