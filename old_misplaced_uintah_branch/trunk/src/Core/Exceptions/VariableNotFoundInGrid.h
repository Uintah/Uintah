
/*
 *  VariableNotFoundInGrid.h: 
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_VARIABLENOTFOUNDINGRID_H
#define UINTAH_EXCEPTIONS_VARIABLENOTFOUNDINGRID_H

#include <SCIRun/Core/Exceptions/Exception.h>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/Exceptions/uintahshare.h>
namespace Uintah {
  
  class UINTAHSHARE VariableNotFoundInGrid : public SCIRun::Exception {
  public:
    VariableNotFoundInGrid(const std::string& varname, long particleID,
			   int matlIndex, const std::string& extramsg, 
                           const char* file, int line);
    VariableNotFoundInGrid(const std::string& varname, SCIRun::IntVector loc,
			   int matlIndex, const std::string& extramsg,
                           const char* file, int line);
    VariableNotFoundInGrid(const std::string& varname,
			   const std::string& extramsg,
                           const char* file, int line);
    VariableNotFoundInGrid(const VariableNotFoundInGrid&);
    virtual ~VariableNotFoundInGrid();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string d_msg;
    VariableNotFoundInGrid& operator=(const VariableNotFoundInGrid&);
  };
  
} // End namespace Uintah

#endif


