/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#include <Core/Disclosure/UnstructuredTypeUtils.h>
#include <Core/Disclosure/UnstructuredTypeDescription.h>

#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <sci_defs/bits_defs.h> // for SCI_32BITS
#include <sci_defs/osx_defs.h>  // for OSX_SNOW_LEOPARD_OR_LATER

#include <cfloat>
#include <climits>

using namespace std;

namespace Uintah {

#if 0
  
#if !defined(STATIC_BUILD)
template<> const string find_type_name(Uintah::long64*)
{
  static const string name = "long64";
  return name;
}
#endif

#endif

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(double*)
{
  static UnstructuredTypeDescription* td;
  if(!td){
    td = scinew UnstructuredTypeDescription( UnstructuredTypeDescription::double_type, "double", true, MPI_DOUBLE );
  }
  return td;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(float*)
{
  static UnstructuredTypeDescription* td;
  if(!td){
    td = scinew UnstructuredTypeDescription( UnstructuredTypeDescription::float_type, "float", true, MPI_FLOAT );
  }
  return td;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(int*)
{
  static UnstructuredTypeDescription* td;
  if(!td){
    td = scinew UnstructuredTypeDescription( UnstructuredTypeDescription::int_type, "int", true, MPI_INT );
  }
  return td;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(short int*)
{
  static UnstructuredTypeDescription* td;
  if(!td){
    td = scinew UnstructuredTypeDescription( UnstructuredTypeDescription::short_int_type, "short int", true, MPI_INT );
  }
  return td;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(FILE**)
{
  static UnstructuredTypeDescription* td;
  if(!td){
    td = scinew UnstructuredTypeDescription( UnstructuredTypeDescription::UnstructuredParticleVariable, "filePointer", true, MPI_BYTE );
  }
  return td;
}

//  const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(long*)
//  {
//     static UnstructuredTypeDescription* td;
//     if(!td){
//        td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::long_type,
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

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(long64*)
{
   static UnstructuredTypeDescription* td;
   if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::long64_type,
				  "long64", true, MPI_LONG_LONG_INT);
   }
   return td;
}


#if !defined( OSX_SNOW_LEOPARD_OR_LATER ) &&  !defined( SCI_32BITS )
const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(long long*)
{
   static UnstructuredTypeDescription* td;
   if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::long64_type,
				  "long64", true, MPI_LONG_LONG_INT);
   }
   return td;
}
#endif

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(bool*)
{
   static UnstructuredTypeDescription* td;
   if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::bool_type,
				  "bool", true, MPI_UB);
   }
   return td;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(unsigned char*)
{
   static UnstructuredTypeDescription* td;
   if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::bool_type,
				  "uchar", true, MPI_UB);
   }
   return td;
}

static MPI_Datatype makeMPI_Point()
{
   ASSERTEQ(sizeof(Point), sizeof(double)*3);
   MPI_Datatype mpitype;
   Uintah::MPI::Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   Uintah::MPI::Type_commit(&mpitype);
   return mpitype;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Point*)
{
   static UnstructuredTypeDescription* td;
   if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::Point,
				  "Point", true, &makeMPI_Point);
   }
   return td;
}

static MPI_Datatype makeMPI_Vector()
{
   ASSERTEQ(sizeof(Vector), sizeof(double)*3);
   MPI_Datatype mpitype;
   Uintah::MPI::Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   Uintah::MPI::Type_commit(&mpitype);
   return mpitype;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Vector*)
{
   static UnstructuredTypeDescription* td;
   if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::Vector,
				  "Vector", true, &makeMPI_Vector);
   }
   return td;
}

static MPI_Datatype makeMPI_IntVector()
{
   ASSERTEQ(sizeof(Vector), sizeof(double)*3);
   MPI_Datatype mpitype;
   Uintah::MPI::Type_vector(1, 3, 3, MPI_INT, &mpitype);
   Uintah::MPI::Type_commit(&mpitype);
   return mpitype;
}

const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(IntVector*)
{
   static UnstructuredTypeDescription* td;
   if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::IntVector,
				  "IntVector", true, &makeMPI_IntVector);
   }
   return td;
}

#if 0  
void fun_getLargeValue(bool* val)
{
  // this should never get called.  It doesn't make sense for a bool
  // to get used in a min or max op
  *val = true;
}

void fun_getSmallValue(bool* val)
{
  // this should never get called.  It doesn't make sense for a bool
  // to get used in a min or max op
  *val = false;
}
void fun_getZeroValue(  double  * val ) { *val = 0; }
#if !defined( SCI_32BITS ) && !defined( OSX_SNOW_LEOPARD_OR_LATER )
void fun_getZeroValue(  long long * val ) { *val = 0; }
#endif
void fun_getZeroValue(  bool    * val ) { *val = false; }
void fun_getZeroValue(  long64  * val ) { *val = 0; }
void fun_getZeroValue(  Vector  * val ) { *val = Vector(0,0,0); }
void fun_getZeroValue(  IntVector  * val ) { *val = IntVector(0,0,0); }

void fun_getLargeValue( long64  * val ) { *val = LONG_MAX; }
void fun_getLargeValue( Vector  * val ) { *val = Vector(DBL_MAX,DBL_MAX,DBL_MAX); }
void fun_getLargeValue( IntVector  * val ) { *val = IntVector(INT_MAX,INT_MAX,INT_MAX); }
void fun_getLargeValue( double  * val ) { *val = DBL_MAX; }

void fun_getSmallValue( long64  * val ) { *val = LONG_MIN; }
void fun_getSmallValue( Vector  * val ) { *val = Vector(-DBL_MAX,-DBL_MAX,-DBL_MAX); }
void fun_getSmallValue( IntVector  * val ) { *val = IntVector(-INT_MAX,-INT_MAX,-INT_MAX); }
void fun_getSmallValue( double  * val ) { *val = -DBL_MAX; }

#endif

} // End namespace Uintah
