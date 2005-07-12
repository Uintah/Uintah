
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <iostream>

using namespace Uintah;

InvalidGrid::InvalidGrid(const std::string& msg, const char* file, int line)
    : d_msg(msg)
{
#ifdef EXCEPTIONS_CRASH
  std::cout << "InvalidGrid Exception: " << file << ":" << line << "\n";
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
