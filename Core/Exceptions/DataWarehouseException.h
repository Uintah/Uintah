
#ifndef UINTAH_HOMEBREW_DataWarehouseException_H
#define UINTAH_HOMEBREW_DataWarehouseException_H

#include <SCICore/Exceptions/Exception.h>
#include <string>

class DataWarehouseException : public SCICore::Exceptions::Exception {
    std::string msg;
public:
    DataWarehouseException(const std::string&);
    virtual std::string message() const;
};

#endif
