
#ifndef UINTAH_HOMEBREW_TypeMismatchException_H
#define UINTAH_HOMEBREW_TypeMismatchException_H

#include <SCICore/Exceptions/Exception.h>
#include <string>

class TypeMismatchException : public SCICore::Exceptions::Exception {
    std::string msg;
public:
    TypeMismatchException(const std::string&);
    TypeMismatchException(const TypeMismatchException&);
    const char* message() const;
    const char* type() const;

private:
    TypeMismatchException& operator=(const TypeMismatchException&);
};

#endif
