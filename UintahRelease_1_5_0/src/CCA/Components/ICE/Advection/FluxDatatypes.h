/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Endian.h> // for other swapbytes() functions.
namespace Uintah {

 /*______________________________________________________________________
 *   different data types 
 *______________________________________________________________________*/ 
  struct fflux { double d_fflux[6]; };          //face flux
  //__________________________________
  // face data
  template <class T> struct facedata {
    T d_data[6];
  };
  
  template<class T>
  MPI_Datatype makeMPI_facedata()
  {
    ASSERTEQ(sizeof(facedata<T>), sizeof(T)*6);
    const TypeDescription* td = fun_getTypeDescription((T*)0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, td->getMPIType(), &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  template<class T>
  const TypeDescription* fun_getTypeDescription(facedata<T>*)
  {
    static TypeDescription* td = 0;
    if(!td){
      //some compilers don't like passing templated function pointers directly
      //across function calls
      MPI_Datatype (*func)() = makeMPI_facedata<T>;
      td = scinew TypeDescription(TypeDescription::Other,
				  "facedata", true, 
				  func);
    }
    return td;
  }
  
  //__________________________________
  // vertex data
  template <class T> struct vertex {
    T d_vrtx[8];
  };
  
  template<class T>
  MPI_Datatype makeMPI_vertex()
  {
    ASSERTEQ(sizeof(vertex<T>), sizeof(T)*8);
    const TypeDescription* td = fun_getTypeDescription((T*)0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 8, 8, td->getMPIType(), &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  template<class T>
  const TypeDescription* fun_getTypeDescription(vertex<T>*)
  {
    static TypeDescription* td = 0;
    if(!td){
      //some compilers don't like passing templated function pointers directly
      //across function calls
      MPI_Datatype (*func)() = makeMPI_vertex<T>;
      td = scinew TypeDescription(TypeDescription::Other,
				  "vertex", true, 
				  func);
    }
    return td;
  }  
  
  const TypeDescription* fun_getTypeDescription(fflux*);

}  // Uintah namespace


//__________________________________
namespace SCIRun {

  template<class T>
  void swapbytes( Uintah::facedata<T>& f) {
    for(int i=0;i<6;i++)
      swapbytes(f.d_data[i]);
  }
  
  template<class T>
  void swapbytes( Uintah::vertex<T>& v) {
    for(int i=0;i<8;i++)
      swapbytes(v.d_vrtx[i]);
  }
  
  void swapbytes( Uintah::fflux& );

}
