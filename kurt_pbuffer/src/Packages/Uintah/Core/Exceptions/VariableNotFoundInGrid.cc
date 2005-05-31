
#include <Packages/Uintah/Core/Exceptions/VariableNotFoundInGrid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>
#include <sstream>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

VariableNotFoundInGrid::VariableNotFoundInGrid(const std::string& varname,
					       long particleID, int matlIndex,
					       const std::string& extramsg)
{
  ostringstream s;
  s << "Particle Variable not found: " << varname;
  
  s << " with particleID (" << particleID << ")";
  if (matlIndex >= 0)
    s << ", material index: " << matlIndex;
  
  if(extramsg != "")
    s << " (" << extramsg << ")";
  d_msg = s.str();
}

VariableNotFoundInGrid::VariableNotFoundInGrid(const std::string& varname,
					       IntVector loc,
					       int matlIndex,
					       const std::string& extramsg)
{
  ostringstream s;
  s << "Grid Variable not found: " << varname;
  
  flush(cerr);
  s << " with location [" << loc.x() << ", " << loc.y() << ", " << loc.z() << "]";
  //s << " with location (" << loc << ")";
  s << ", material index: " << matlIndex;
  
  if(extramsg != "")
    s << " (" << extramsg << ")";
  d_msg = s.str();
}

VariableNotFoundInGrid::VariableNotFoundInGrid(const std::string& varname,
					       const std::string& extramsg)
{
  ostringstream s;
  s << "Variable not found: " << varname;
  if(extramsg != "")
    s << " (" << extramsg << ")";
  d_msg = s.str();
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
