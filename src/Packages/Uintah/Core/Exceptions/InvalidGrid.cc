
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <iostream>

using namespace Uintah;

InvalidGrid::InvalidGrid(const std::string& msg)
    : d_msg(msg)
{
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
