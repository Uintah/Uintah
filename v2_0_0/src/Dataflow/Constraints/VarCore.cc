/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  VarCore.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Dataflow/Constraints/VarCore.h>
#include <iostream>
using std::ostream;
#include <stdio.h>

namespace SCIRun {

int
VarCore::operator==( const VarCore& c ) const
{
   if (vartype == c.vartype) {
      if ((vartype == PointVar) && (pointvalue == c.pointvalue)) return 1;
      else if (realvalue == c.realvalue) return 1;
   }
   return 0;
}


int
VarCore::operator==( const Point& p ) const
{
   if ((vartype == PointVar) && (pointvalue == p))
      return 1;
   else
      return 0;
}


int
VarCore::operator==( const double r ) const
{
   if ((vartype == RealVar) && (realvalue == r))
      return 1;
   else
      return 0;
}


VarCore&
VarCore::operator+=( const Vector& v )
{
   ASSERT(vartype == PointVar);
   pointvalue += v;
   return *this;
}


VarCore&
VarCore::operator+=( const double r )
{
   ASSERT(vartype == RealVar);
   realvalue += r;
   return *this;
}


ostream& operator<<( ostream& os, VarCore& c ) {
   if (c.vartype==VarCore::PointVar) os << c.pointvalue;
   else os << c.realvalue;
   return os;
}

} // End namespace SCIRun



