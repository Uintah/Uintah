//static char *id="@(#) $Id$";

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

#include <PSECore/Constraints/VarCore.h>
#include <iostream>
using std::ostream;
#include <stdio.h>

namespace PSECore {
namespace Constraints {

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

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:17  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/08 02:26:40  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/17 06:38:20  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:56  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//


