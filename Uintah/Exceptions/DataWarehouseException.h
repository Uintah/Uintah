#ifndef UINTAH_HOMEBREW_DataWarehouseException_H
#define UINTAH_HOMEBREW_DataWarehouseException_H

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace Uintah {
namespace Exceptions {

/**************************************

CLASS
   DataWarehouseException
   
   Short description...

GENERAL INFORMATION

   DataWarehouseException.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Exception_DataWarehouse

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class DataWarehouseException : public SCICore::Exceptions::Exception {
    std::string msg;
public:
    DataWarehouseException(const std::string&);
    virtual const char* message() const;
    virtual const char* type() const;
};

} // end namespace Exception
} // end namespace Uintah

#endif
