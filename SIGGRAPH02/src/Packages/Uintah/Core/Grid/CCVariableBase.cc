#include <Packages/Uintah/Core/Grid/CCVariableBase.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>

using namespace Uintah;
using namespace SCIRun;

extern Mutex MPITypeLock;

CCVariableBase::~CCVariableBase()
{
}

CCVariableBase::CCVariableBase()
{
}

void CCVariableBase::getMPIBuffer(BufferInfo& buffer,
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
