/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

/////////////////////////////////////////
//
//  Note, the 'Type' for BoundCond is usually 'double', 'vector', or 'string' ('string', is just a dummy
//  to hold a value of 'no value' in the case where the BoundCond does not have an associated <value> tag).
//

template <class Type>
class BoundCond : public BoundCondBase
{
public:

  BoundCond( const std::string & var_name,
             const std::string & type,
             const Type        & value, 
             const std::string & face_label,
             const std::string & functor_name,
                   int           matl_id ) :
    BoundCondBase( var_name, type, face_label, functor_name, matl_id )
  {
    d_value = value;
  }

  virtual ~BoundCond() {}

  virtual BoundCond* clone()
  {
    return scinew BoundCond<Type>( *this );
  }

  virtual void debug() const;

  Type getValue() const { return d_value; }

protected:

   Type d_value;

private:

  // Disallow creation of empty BoundConds.
  BoundCond() {}


};

#if 0
Tonys stuff... put back in if we need it... 

   BoundCond(string var_name,string type)
     {
       d_variable = var_name;
       d_type__NEW = type;
       d_value = NoValue();
       d_face_label = "none";
       d_functor_name = "none";
     }

   BoundCond(string var_name)
     {
       d_variable = var_name;
       d_type__NEW = "";
       d_value = NoValue();
       d_face_label = "none";
       d_functor_name = "none";
     }
#endif

} // End namespace Uintah

#endif
