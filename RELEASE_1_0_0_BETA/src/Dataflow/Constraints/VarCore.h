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


#ifndef SCI_project_VarCore_h
#define SCI_project_VarCore_h 1

#include <Dataflow/share/share.h>
#include <Dataflow/Constraints/manifest.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {


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
   friend PSECORESHARE std::ostream& operator<<( std::ostream& os, VarCore& c );
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
      //return 0;
   }
}

} // End namespace SCIRun

#endif
