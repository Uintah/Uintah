#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Parallel/BufferInfo.h>

#include <SCIRun/Core/Geometry/IntVector.h>
#include <SCIRun/Core/Exceptions/InternalError.h>
#include <SCIRun/Core/Thread/Mutex.h>

using namespace Uintah;
using namespace SCIRun;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllimport)
#else
#define UINTAHSHARE
#endif

extern UINTAHSHARE Mutex MPITypeLock;

void GridVariableBase::getMPIBuffer(BufferInfo& buffer,
	  			    const IntVector& low, const IntVector& high)
{
  const TypeDescription* td = virtualGetTypeDescription()->getSubType();
  MPI_Datatype basetype=td->getMPIType();
  IntVector l, h, s, strides, dataLow;
  getSizes(l, h, dataLow, s, strides);

  IntVector off = low - dataLow;
  char* startbuf = (char*)getBasePointer();
  startbuf += strides.x()*off.x()+strides.y()*off.y()+strides.z()*off.z();
  IntVector d = high-low;
  MPI_Datatype type1d;
 MPITypeLock.lock();
  MPI_Type_hvector(d.x(), 1, strides.x(), basetype, &type1d);
  using namespace std;
  MPI_Datatype type2d;
  MPI_Type_hvector(d.y(), 1, strides.y(), type1d, &type2d);
  MPI_Type_free(&type1d);
  MPI_Datatype type3d;
  MPI_Type_hvector(d.z(), 1, strides.z(), type2d, &type3d);
  MPI_Type_free(&type2d);
  MPI_Type_commit(&type3d);
 MPITypeLock.unlock();  
  buffer.add(startbuf, 1, type3d, true);
}
