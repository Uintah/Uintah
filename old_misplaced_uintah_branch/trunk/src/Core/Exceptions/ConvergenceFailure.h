
/*
 *  ConvergenceFailure.h: 
 *
 *  Written by:
 *   John Schmidt
 *   Department of Mechanical Engineering
 *   University of Utah
 *   Nov 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_ConvergenceFailure_H
#define UINTAH_EXCEPTIONS_ConvergenceFailure_H

#include <SCIRun/Core/Exceptions/Exception.h>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/Exceptions/uintahshare.h>
namespace Uintah {
  using namespace std;
  class UINTAHSHARE ConvergenceFailure : public SCIRun::Exception {
  public:
    ConvergenceFailure(const string& msg,
		       int numiterations, double final_residual,
		       double target_residual,
                       char* file, int line);
    ConvergenceFailure(const ConvergenceFailure&);
    virtual ~ConvergenceFailure();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string d_msg;
    ConvergenceFailure& operator=(const ConvergenceFailure&);
  };
  
} // End namespace Uintah

#endif


