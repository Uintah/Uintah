
#include <Uintah/Grid/UnknownVariable.h>
#include <Uintah/Grid/Patch.h>
#include <sstream>

using namespace Uintah;
using namespace std;

UnknownVariable::UnknownVariable(const std::string& varname,
				 const Patch* patch,
				 int matlIndex, const std::string& extramsg)
{
   ostringstream s;
   s << "Unknown variable: " << varname;

   if (patch != NULL) {
      s << " on patch " << patch->getID()
        << "(" << patch->toString() << ")";
   }
   if (matlIndex >= 0)
      s << ", material index: " << matlIndex;

   if(extramsg != "")
      s << " (" << extramsg << ")";
   d_msg = s.str();
}

UnknownVariable::UnknownVariable(const std::string& varname,
				 const std::string& extramsg)
{
   ostringstream s;
   s << "Unknown variable: " << varname;
   if(extramsg != "")
      s << " (" << extramsg << ")";
   d_msg = s.str();
}

UnknownVariable::UnknownVariable(const UnknownVariable& copy)
    : d_msg(copy.d_msg)
{
}

UnknownVariable::~UnknownVariable()
{
}

const char* UnknownVariable::message() const
{
    return d_msg.c_str();
}

const char* UnknownVariable::type() const
{
    return "Uintah::Exceptions::UnknownVariable";
}

//
// $Log$
// Revision 1.1  2000/12/23 00:38:44  witzel
// moved from Uintah/Exceptions
//
// Revision 1.6  2000/12/06 23:41:39  witzel
// Changed UnknownVariable constructor to take Patch* instead
// of patch id and string and allow this pointer to be NULL.
//
// Revision 1.5  2000/09/26 21:32:24  dav
// Formatting
//
// Revision 1.4  2000/06/19 22:36:32  sparker
// Improved message for Unknown variable
//
//

