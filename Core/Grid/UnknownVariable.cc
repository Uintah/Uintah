
#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <sstream>
#include <iostream>

using namespace Uintah;
using namespace std;

UnknownVariable::UnknownVariable(const std::string& varname, int dwid,
				 const Patch* patch, int matlIndex,
				 const std::string& extramsg,
                                 const char* file,
                                 int line)
{
   ostringstream s;
   s << "Unknown variable: " << varname;

   s << " requested from DW " << dwid;
   if (patch != NULL) {
      s << " on patch " << patch->getID()
        << "(" << patch->toString() << ")";
   }
   if (matlIndex >= 0)
      s << ", material index: " << matlIndex;

   if(extramsg != "")
      s << " (" << extramsg << ")";
   d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
   std::cout << "An UnknownVariable exception was thrown.\n";
   std::cout << file << ":" << line << "\n";
   std::cout << d_msg << "\n";
#endif
}

UnknownVariable::UnknownVariable(const std::string& varname, int dwid,
				 const Level* level, int matlIndex,
				 const std::string& extramsg,
                                 const char* file,
                                 int line)
{
   ostringstream s;
   s << "Unknown variable: " << varname;

   s << " requested from DW " << dwid;
   if (level != NULL) {
     s << " on level " << level->getIndex();
   }
   if (matlIndex >= 0)
      s << ", material index: " << matlIndex;

   if(extramsg != "")
      s << " (" << extramsg << ")";
   d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
   std::cout << "An UnknownVariable exception was thrown.\n";
   std::cout << file << ":" << line << "\n";
   std::cout << d_msg << "\n";
#endif
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
