
/*
 *  DeniedAccess.h: 
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef UINTAH_COMPONENTS_SCHEDULERS_DENIEDACCESS_H
#define UINTAH_COMPONENTS_SCHEDULERS_DENIEDACCESS_H

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {

   class DeniedAccess : public Exception {
   public:
     DeniedAccess(const VarLabel* label, const Task* task, int matlIndex,
		  const Patch* patch, string dependency, string accessType);
     DeniedAccess(const DeniedAccess& copy);
     virtual ~DeniedAccess() {}
     virtual const char* message() const;
     virtual const char* type() const;
   protected:
   private:
     DeniedAccess& operator=(const DeniedAccess& copy);
     const VarLabel* label_;
     const Task* task_;
     int matlIndex_;     
     const Patch* patch_;
     string d_msg;
   };

} // End namespace Uintah

#endif


