/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
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
   std::ostringstream s;
   s << "An UnknownVariable exception was thrown.\n";
   s << file << ":" << line << "\n";

   s << "Unknown variable: " << varname;

   if (dwid!=-1) s << " requested from DW " << dwid;
   if (patch != NULL) {
      s << ", Level "<< patch->getLevel()->getIndex()
        << ", patch " << patch->getID()
        << "(" << patch->toString() << ")";
   }
   
   s << ", material index: " << matlIndex;

   if(extramsg != "")
      s << " (" << extramsg << ")";
   d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
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

   if (dwid!=-1) s << " requested from DW " << dwid;
   if (level != NULL) {
     s << " on level " << level->getIndex();
   }
   
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
