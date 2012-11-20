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

#ifndef UINTAH_GRID_BoundCondBase_H
#define UINTAH_GRID_BoundCondBase_H

#include <string>

#include <iostream> // FIXME: for debugging cout

namespace Uintah {
   
/**************************************

CLASS
   BoundCondBase
   
   
GENERAL INFORMATION

   BoundCondBase.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   BoundCondBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class BoundCondBase
{
public:
  
  BoundCondBase( const std::string & var_name,
                 const std::string & type,
                 const std::string & face_label,
                 const std::string & functor_name,
                       int           matl_id ) :
    d_variable( var_name ), d_type( type ), d_face_label( face_label ), d_functor_name( functor_name ), d_matl_id( matl_id )
  {
  }

  virtual ~BoundCondBase() { std::cout << "in ~BoundCondBase() for " << this << "\n"; }

  const std::string & getVariable() const { return d_variable; }
  const std::string & getType() const     { return d_type; }
        int           getMatl() const     { return d_matl_id; }

  const std::string & getBCFaceName() const  { return d_face_label; }
  const std::string & getFunctorName() const { return d_functor_name; }

  virtual void debug() const = 0;
  
protected:
  std::string d_variable;     // Eg: Pressure, Density, etc
  std::string d_type;         // Eg: Dirichlet, Neumann, etc
  std::string d_face_label;   // The user specified name of the bc face: left-wall, ox-inlet, etc...
  std::string d_functor_name; // The name of a functor to be applied on this boundary.
  int         d_matl_id;      // Material Id.  -1 == "all materials"
  
};

} // End namespace Uintah

#endif
