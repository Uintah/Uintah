
/*
 *  IncorrectAllocation.h: 
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef UINTAH_COMPONENTS_SCHEDULERS_INCORRECT_ALLOCATION_H
#define UINTAH_COMPONENTS_SCHEDULERS_INCORRECT_ALLOCATION_H

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using SCIRun::Exception;

  class IncorrectAllocation : public Exception {
  public:
    IncorrectAllocation(const VarLabel* expectedLabel,
			const VarLabel* actualLabel);
    IncorrectAllocation(const IncorrectAllocation& copy);
    virtual ~IncorrectAllocation() {}

    static string makeMessage(const VarLabel* expectedLabel,
			      const VarLabel* actualLabel);
     
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    IncorrectAllocation& operator=(const IncorrectAllocation& copy);
    const VarLabel* expectedLabel_;
    const VarLabel* actualLabel_;
    string d_msg;
  };

} // End namespace Uintah

#endif


