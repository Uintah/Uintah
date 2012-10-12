/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>
#include <sstream>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;


VariableNotFoundInGrid::VariableNotFoundInGrid(const std::string& varname,
					       long particleID, int matlIndex,
					       const std::string& extramsg,
                                               const char* file, int line)
{
  ostringstream s;
  s << "A VariableNotFoundInGrid exception was thrown.\n"
    << file << ":" << line << "\n";

  s << "Particle Variable not found: " << varname;
  
  s << " with particleID (" << particleID << ")";
  if (matlIndex >= 0)
    s << ", material index: " << matlIndex;
  
  if(extramsg != "")
    s << " (" << extramsg << ")";
  d_msg = s.str();
  
#ifdef EXCEPTIONS_CRASH
  cout << d_msg << "\n";
#endif
}

VariableNotFoundInGrid::VariableNotFoundInGrid(const std::string& varname,
					       IntVector loc,
					       int matlIndex,
					       const std::string& extramsg,
                                               const char* file, int line)
{
  std::ostringstream s;
  s << "Grid Variable not found: " << varname;
  
  std::flush(std::cerr);
  s << " with location [" << loc.x() << ", " << loc.y() << ", " << loc.z() << "]";
  //s << " with location (" << loc << ")";
  s << ", material index: " << matlIndex;
  
  if(extramsg != "")
    s << " (" << extramsg << ")";
  d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
  std::cout << "A VariableNotFoundInGrid exception was thrown.\n";
  std::cout << file << ":" << line << "\n";
  std::cout << d_msg;
#endif
}

VariableNotFoundInGrid::VariableNotFoundInGrid(const std::string& varname,
					       const std::string& extramsg,
                                               const char* file, int line)
{
  ostringstream s;
  s << "Variable not found: " << varname;
  if(extramsg != "")
    s << " (" << extramsg << ")";
  d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
  cout << "A VariableNotFoundInGrid exception was thrown.\n";
  cout << file << ":" << line << "\n";
  cout << d_msg;
#endif
}

VariableNotFoundInGrid::VariableNotFoundInGrid(const VariableNotFoundInGrid& copy)
  : d_msg(copy.d_msg)
{
}

VariableNotFoundInGrid::~VariableNotFoundInGrid()
{
}

const char* VariableNotFoundInGrid::message() const
{
  return d_msg.c_str();
}

const char* VariableNotFoundInGrid::type() const
{
  return "Packages/Uintah::Exceptions::VariableNotFoundInGrid";
}
