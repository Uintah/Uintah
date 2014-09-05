
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <iostream>
#include <sstream>
using namespace Uintah;

InvalidGrid::InvalidGrid(const std::string& msg, const char* file, int line)
    : d_msg(msg)
{
  std::ostringstream s;
  s << "InvalidGrid Exception: " << file << ":" << line << "\n" << d_msg;
  d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
  std::cout << d_msg << "\n";
#endif
}

InvalidGrid::InvalidGrid(const InvalidGrid& copy)
    : d_msg(copy.d_msg)
{
}

InvalidGrid::~InvalidGrid()
{
}

const char* InvalidGrid::message() const
{
    return d_msg.c_str();
}

const char* InvalidGrid::type() const
{
    return "Uintah::Exceptions::InvalidGrid";
}
