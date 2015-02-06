/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_GRID_BoundCond_H
#define UINTAH_GRID_BoundCond_H

#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <string>

namespace Uintah {

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

   BoundCond( const std::string var_name,
              const std::string type,
              const T value,
              const std::string face_label,
              const BoundCondBase::BoundCondValueTypeEnum val_type)
     {
       d_variable     = var_name;
       d_type         = type;
       d_value        = value;
       d_face_label   = face_label;
       d_value_type   = val_type;
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

   BoundCond(std::string var_name,std::string type)
     {
       d_variable = var_name;
       d_type = type;
       d_value = NoValue();
       d_face_label = "none";
       d_value_type = BoundCondBase::UNKNOWN_TYPE;
     };

   BoundCond(std::string var_name)
     {
       d_variable = var_name;
       d_type = "";
       d_value = NoValue();
       d_face_label = "none";
       d_value_type = BoundCondBase::UNKNOWN_TYPE;
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
