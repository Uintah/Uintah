
#include "DataWarehouseException.h"

DataWarehouseException::DataWarehouseException(const std::string& msg)
    : msg(msg)
{
}

std::string DataWarehouseException::message() const
{
    return msg;
}
