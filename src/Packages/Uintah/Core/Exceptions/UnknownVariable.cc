
#include <Uintah/Exceptions/UnknownVariable.h>
#include <sstream>

using namespace Uintah;
using namespace std;

UnknownVariable::UnknownVariable(const std::string& varname,
				 int patchNumber, const std::string& patch,
				 int matlIndex, const std::string& extramsg)
{
   ostringstream s;
   s << "Unknown variable: " << varname << " on patch " << patchNumber << "(" << patch << ")" << ", material index: " << matlIndex;
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
// Revision 1.4  2000/06/19 22:36:32  sparker
// Improved message for Unknown variable
//
//

