/*
 *  Interpolation.h: interface for interpolation classes
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef SCI_INTERPOLATION_H__
#define SCI_INTERPOLATION_H__

#include <SCICore/Containers/Array1.h>
#include <SCICore/share/share.h>

namespace SCICore{
namespace Math{

class SCICORESHARE Interpolation {
protected:
    bool cldata;   // flag to attempt to "clean" data if needed
public:
    // static function for "on-fly" interpolations - to be implemented
    // .....				      
    virtual double get_value(double)=0;	
    virtual ~Interpolation() {};
    inline void clean_data(bool);
};

inline void Interpolation::clean_data(bool i) { 
  cldata=i; 
}

} // Math
} // SCICore

#endif //SCI_INTERPOLATION_H__
