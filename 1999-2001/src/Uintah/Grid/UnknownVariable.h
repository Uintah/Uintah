
// $Id$

/*
 *  UnknownVariable.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Uintah_Exceptions_UnknownVariable_h
#define Uintah_Exceptions_UnknownVariable_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace Uintah {
class Patch;

   class UnknownVariable : public SCICore::Exceptions::Exception {
   public:
      UnknownVariable(const std::string& varname, const Patch* patch,
		      int matlIndex, const std::string& extramsg = "");
      UnknownVariable(const std::string& varname,
		      const std::string& extramsg);
      UnknownVariable(const UnknownVariable&);
      virtual ~UnknownVariable();
      virtual const char* message() const;
      virtual const char* type() const;
   protected:
   private:
      std::string d_msg;
      UnknownVariable& operator=(const UnknownVariable&);
   };
}

#endif

//
// $Log$
// Revision 1.1  2000/12/23 00:38:44  witzel
// moved from Uintah/Exceptions
//
// Revision 1.5  2000/12/06 23:41:39  witzel
// Changed UnknownVariable constructor to take Patch* instead
// of patch id and string and allow this pointer to be NULL.
//
// Revision 1.4  2000/09/26 21:32:24  dav
// Formatting
//
// Revision 1.3  2000/06/19 22:36:32  sparker
// Improved message for Unknown variable
//
// Revision 1.2  2000/04/26 06:48:43  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/11 07:10:46  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
//

