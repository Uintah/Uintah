
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


/*#include <stdio.h>
#include <string.h>*/
#include <Constraints/VarCore.h>


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
VarCore::operator==( const Real r ) const
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
VarCore::operator+=( const Real r )
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


