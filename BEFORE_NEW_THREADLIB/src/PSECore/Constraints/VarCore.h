
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


#ifndef SCI_project_VarCore_h
#define SCI_project_VarCore_h 1

#include <PSECore/share/share.h>
#include <PSECore/Constraints/manifest.h>
#include <SCICore/Geometry/Point.h>

namespace PSECore {
namespace Constraints {

using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

// VarCore is the data of a Variable.  It implements the polymorphism.
class BaseVariable;
class PSECORESHARE VarCore {
public:
   enum VarType { PointVar, RealVar };
   // This controls operator= with different VarTypes.
   enum Rigidity { NonRigid, Rigid };

   VarCore(const Point& p, Rigidity rigid=NonRigid)
   : vartype(PointVar), pointvalue(p), rigidity(rigid) {}
   VarCore(const Real& r, Rigidity rigid=NonRigid)
   : vartype(RealVar), realvalue(r), rigidity(rigid) {}

   inline int isPoint() const { return (vartype==PointVar); }
   inline int isReal() const { return (vartype==RealVar); }

   inline const Point& point() const { ASSERT(vartype==PointVar); return pointvalue; }
   inline operator Point() const { ASSERT(vartype==PointVar); return pointvalue; }
   inline Real real() const { ASSERT(vartype==RealVar); return realvalue; }
   inline operator Real() const { ASSERT(vartype==RealVar); return realvalue; }

   // CAN assign different types.
   inline VarCore& operator=( const VarCore& c );
   inline VarCore& operator=( const Point& p );
   inline VarCore& operator=( const Real r );
   int operator==( const VarCore& c ) const;
   int operator==( const Point& p ) const;
   int operator==( const Real r ) const;

   VarCore& operator+=( const Vector& v );
   VarCore& operator+=( const Real r );

   inline int epsilonequal( const Real Epsilon, const VarCore& v );
   friend PSECORESHARE ostream& operator<<( ostream& os, VarCore& c );
private:
   VarType vartype;
   Point pointvalue;
   Real realvalue;
   Rigidity rigidity;
};


inline VarCore&
VarCore::operator=( const VarCore& c )
{
   if (rigidity == Rigid) {
      ASSERT(vartype == c.vartype);
   } else {
      vartype = c.vartype;
   }
   if (vartype == PointVar) pointvalue = c.pointvalue;
   else realvalue = c.realvalue;
   return *this;
}


inline VarCore&
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


inline VarCore&
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


inline int
VarCore::epsilonequal( const Real Epsilon, const VarCore& v )
{
   if (isPoint() && v.isPoint())
      return ((RealAbs(pointvalue.x()-v.pointvalue.x()) < Epsilon)
	      && (RealAbs(pointvalue.y()-v.pointvalue.y()) < Epsilon)
	      && (RealAbs(pointvalue.z()-v.pointvalue.z()) < Epsilon));
   else if (isReal() && v.isReal())
      return (RealAbs(realvalue-v.realvalue) < Epsilon);
   else {
      ASSERTFAIL("Can't compare PointVariable with RealVariable!!");
      return 0;
   }
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/08/26 23:57:03  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.3  1999/08/18 20:20:18  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:38:21  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:57  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:08  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif
