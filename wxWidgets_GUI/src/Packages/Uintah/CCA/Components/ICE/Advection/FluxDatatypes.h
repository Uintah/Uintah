#include <mpi.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Util/FancyAssert.h>
namespace Uintah {

 /*______________________________________________________________________
 *   different data types 
 *______________________________________________________________________*/ 
  struct fflux { double d_fflux[6]; };          //face flux
  struct eflux { double d_eflux[12]; };         //edge flux
  struct cflux { double d_cflux[8]; };          //corner flux

  //__________________________________
  // face data
  template <class T> struct facedata {
    T d_data[6];
  };
  
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                // This turns off SGI compiler warning.
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
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif
  
  template<class T>
  const TypeDescription* fun_getTypeDescription(facedata<T>*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "facedata", true, 
				  &makeMPI_facedata<T>);
    }
    return td;
  }
  
  //__________________________________
  // vertex data
  template <class T> struct vertex {
    T d_vrtx[8];
  };
  
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                // This turns off SGI compiler warning.
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
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif
  
  template<class T>
  const TypeDescription* fun_getTypeDescription(vertex<T>*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "vertex", true, 
				  &makeMPI_vertex<T>);
    }
    return td;
  }  
  
  const TypeDescription* fun_getTypeDescription(fflux*);    
  const TypeDescription* fun_getTypeDescription(eflux*);
  const TypeDescription* fun_getTypeDescription(cflux*); 

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
  void swapbytes( Uintah::eflux& );
  void swapbytes( Uintah::cflux& );

}
