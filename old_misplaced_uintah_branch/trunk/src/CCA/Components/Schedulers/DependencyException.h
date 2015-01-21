
/*
 *  DependencyException.h
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef UINTAH_COMPONENTS_SCHEDULERS_DEPENDENCY_EXCEPTION_H
#define UINTAH_COMPONENTS_SCHEDULERS_DEPENDENCY_EXCEPTION_H

#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>
#include <SCIRun/Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using SCIRun::Exception;

  class DependencyException : public Exception {
  public:
    DependencyException(const Task* task, const VarLabel* label,
			int matlIndex, const Patch* patch,
			string has, string needs,
                        const char* file, int line);
    DependencyException(const DependencyException& copy);
    virtual ~DependencyException() {}

    static string
    makeMessage(const Task* task, const VarLabel* label, int matlIndex,
		const Patch* patch, string has, string needs);
     
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    DependencyException& operator=(const DependencyException& copy);
    const Task* task_;
    const VarLabel* label_;
    int matlIndex_;
    const Patch* patch_;
    string d_msg;
  };

} // End namespace Uintah

#endif


