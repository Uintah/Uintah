
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace SCIRun {

using std::string;

template<> const string find_type_name(long64*)
{
  static const string name = "long64";
  return name;
}

const TypeDescription* get_type_description(long64*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("long64", "builtin", "builtin");
  }
  return td;
}

} // namespace SCIRun


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

const TypeDescription* fun_getTypeDescription(short int*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::short_int_type,
				  "short int", true, MPI_INT);
   }
   return td;
}

//  const TypeDescription* fun_getTypeDescription(long*)
//  {
//     static TypeDescription* td;
//     if(!td){
//        td = scinew TypeDescription(TypeDescription::long_type,
//  				  "long", true,
//  #ifdef SCI_64BITS
//        MPI_LONG_LONG_INT
//  #else
//        MPI_LONG
//  #endif
//        );
//     }
//     return td;
//  }

const TypeDescription* fun_getTypeDescription(long64*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::long64_type,
				  "long64", true, MPI_LONG_LONG_INT);
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

const TypeDescription* fun_getTypeDescription(unsigned char*)
{
   static TypeDescription* td;
   if(!td){
      td = scinew TypeDescription(TypeDescription::bool_type,
				  "uchar", true, MPI_UB);
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
