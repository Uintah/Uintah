
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

#include <Core/Exceptions/Exception.h>
#include <Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  
  class VariableNotFoundInGrid : public SCIRun::Exception {
  public:
    VariableNotFoundInGrid(const std::string& varname, long particleID,
			   int matlIndex, const std::string& extramsg = "");
    VariableNotFoundInGrid(const std::string& varname, SCIRun::IntVector loc,
			   int matlIndex, const std::string& extramsg = "");
    VariableNotFoundInGrid(const std::string& varname,
			   const std::string& extramsg);
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


