
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
epsilonequal( const Real Epsilon, const VarCore& p1, const VarCore& p2 )
{
   if (p1.isPoint() && p2.isPoint())
      return ((RealAbs(p1.pointvalue.x()-p2.pointvalue.x()) < Epsilon)
	      && (RealAbs(p1.pointvalue.y()-p2.pointvalue.y()) < Epsilon)
	      && (RealAbs(p1.pointvalue.z()-p2.pointvalue.z()) < Epsilon));
   else if (p1.isReal() && p2.isReal())
      return (RealAbs(p1.realvalue-p2.realvalue) < Epsilon);
   else {
      ASSERT(!"Can't compare PointVariable with RealVariable!!");
      return 0;
   }
}


VarCore&
VarCore::operator=( const VarCore& c )
{
   if (rigidity == Rigid) {
      ASSERT(vartype == c.vartype);
      if (vartype == PointVar) pointvalue = c.pointvalue;
      else realvalue = c.realvalue;
   } else {
      vartype = c.vartype;
      if (vartype == PointVar) pointvalue = c.pointvalue;
      else realvalue = c.realvalue;
   }
   return *this;
}


VarCore&
VarCore::operator=( const Point& p )
{
   if (rigidity == Rigid) {
      ASSERT(vartype == PointVar);
   } else {
      vartype = PointVar;
   }
   pointvalue = p;
   return *this;
}


VarCore&
VarCore::operator=( const Real r )
{
   if (rigidity == Rigid) {
      ASSERT(vartype == RealVar);
   } else {
      vartype = RealVar;
   }
   realvalue = r;
   return *this;
}


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


