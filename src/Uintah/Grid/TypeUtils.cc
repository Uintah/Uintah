
#include <Uintah/Grid/TypeUtils.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Util/FancyAssert.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Malloc/Allocator.h>
using namespace SCICore::Geometry;

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {

const TypeDescription* fun_getTypeDescription(double*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::double_type,
				  "double", true);
   }
   return td;
}

const TypeDescription* fun_getTypeDescription(int*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::int_type,
				  "int", true);
   }
   return td;
}

const TypeDescription* fun_getTypeDescription(bool*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::bool_type,
				  "bool", true);
   }
   return td;
}

const TypeDescription* fun_getTypeDescription(Point*)
{
   static TypeDescription* td;
   if(!td){
      ASSERTEQ(sizeof(Point), sizeof(double)*3);
      td = scinew TypeDescription(TypeDescription::Point,
				  "Point", true);
   }
   return td;
}

const TypeDescription* fun_getTypeDescription(Vector*)
{
   static TypeDescription* td;
   if(!td){
      ASSERTEQ(sizeof(Vector), sizeof(double)*3);
      td = scinew TypeDescription(TypeDescription::Vector,
				  "Vector", true);
   }
   return td;
}

} // End namespace Uintah

//
// $Log$
// Revision 1.2  2000/05/30 20:19:35  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.1  2000/05/20 08:09:29  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//

