
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
				  "double", true, MPI_DOUBLE);
   }
   return td;
}

const TypeDescription* fun_getTypeDescription(int*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::int_type,
				  "int", true, MPI_INT);
   }
   return td;
}

const TypeDescription* fun_getTypeDescription(long*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::long_type,
				  "long", true,
#ifdef SCI_64BITS
      MPI_LONG_LONG_INT
#else
      MPI_LONG
#endif
      );
   }
   return td;
}

const TypeDescription* fun_getTypeDescription(bool*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::bool_type,
				  "bool", true, MPI_UB);
   }
   return td;
}

static MPI_Datatype makeMPI_Point()
{
   ASSERTEQ(sizeof(Point), sizeof(double)*3);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(Point*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Point,
				  "Point", true, &makeMPI_Point);
   }
   return td;
}

static MPI_Datatype makeMPI_Vector()
{
   ASSERTEQ(sizeof(Vector), sizeof(double)*3);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(Vector*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Vector,
				  "Vector", true, &makeMPI_Vector);
   }
   return td;
}

} // End namespace Uintah

//
// $Log$
// Revision 1.5  2000/09/26 17:07:40  sparker
// Need to commit MPI types
//
// Revision 1.4  2000/07/27 22:39:51  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.3  2000/06/02 17:22:14  guilkey
// Added long_type to the the TypeDescription and TypeUtils.
//
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

