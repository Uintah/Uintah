#ifndef UINTAH_GRID_BoundCond_H
#define UINTAH_GRID_BoundCond_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
 using namespace SCIRun;
   
/**************************************

CLASS
   BoundCond
   
   
GENERAL INFORMATION

   BoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

 class NoValue {

 public:
   NoValue() {};
   ~NoValue() {};
 };

 template <class T>  class BoundCond : public BoundCondBase {
 public:
   BoundCond() {};

   BoundCond(string var_name, string type, T value) 
     {
       d_variable = var_name;
       d_type__NEW = type;
       d_value = value;
     };
   virtual ~BoundCond() {};
   virtual BoundCond* clone()
   {
     return scinew BoundCond(*this);
   };

   T getValue() const { return d_value;}; 

   
 protected:
   T d_value;

 };


 template <> class BoundCond<NoValue> : public BoundCondBase {


 public:

   BoundCond(string var_name,string type)
     {
       d_variable = var_name;
       d_type__NEW = type;
       d_value = NoValue();
     };

   BoundCond(string var_name)
     {
       d_variable = var_name;
       d_type__NEW = "";
       d_value = NoValue();
     };

   virtual BoundCond* clone()
   {
     return scinew BoundCond(*this);
   };

   
 protected:
   NoValue d_value;

 };
 
} // End namespace Uintah



#endif
