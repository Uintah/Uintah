/* REFERENCED */
static char *id="@(#) $Id$";

#include "DataWarehouseException.h"

namespace Uintah {
namespace Exceptions {

DataWarehouseException::DataWarehouseException(const std::string& msg)
    : msg(msg)
{
}

DataWarehouseException::DataWarehouseException(const DataWarehouseException& copy)
    : msg(copy.msg)
{
}

const char* DataWarehouseException::message() const
{
    return msg.c_str();
}

const char* DataWarehouseException::type() const
{
    return "Uintah::Exceptions::DataWarehouseException";
}

} // end namespace Exception
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/03/23 20:42:19  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.4  2000/03/23 10:30:29  sparker
// Update to use new Exception base class
//
// Revision 1.3  2000/03/22 23:41:24  sparker
// Working towards getting arches to compile/run
//
// Revision 1.2  2000/03/17 18:45:40  dav
// fixed a few more namespace problems
//
//
