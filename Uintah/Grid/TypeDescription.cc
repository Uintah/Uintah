
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Util/Assert.h>
#include <sci_defs.h>
#include <map>
#include <iostream>

using namespace Uintah;
using namespace std;
using namespace SCICore::Exceptions;

static map<string, const TypeDescription*>* types = 0;

TypeDescription::TypeDescription(Type type, const std::string& name,
				 bool isFlat, MPI_Datatype (*mpitypemaker)())
   : d_type(type), d_subtype(0), d_name(name), d_isFlat(isFlat),
     d_mpitype(-1), d_mpitypemaker(mpitypemaker), d_maker(0)
{
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 bool isFlat, MPI_Datatype mpitype)
   : d_type(type), d_subtype(0), d_name(name), d_isFlat(isFlat),
     d_mpitype(mpitype), d_mpitypemaker(0), d_maker(0)
{
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 Variable* (*maker)(),
				 const TypeDescription* subtype)
   : d_type(type), d_subtype(subtype), d_name(name), d_isFlat(false),
     d_mpitype(-2), d_mpitypemaker(0), d_maker(maker)
{
}

string TypeDescription::getName() const
{
   if(d_subtype) {
      return d_name+"<"+d_subtype->getName()+">";
   } else {
      return d_name;
   }
}

const TypeDescription* TypeDescription::lookupType(const std::string& t)
{
  if(!types)
    types=scinew map<string, const TypeDescription*>;   
  
  map<string, const TypeDescription*>::iterator iter = types->find(t);
   if(iter == types->end())
      return 0;
   return iter->second;
}

TypeDescription::Register::Register(const TypeDescription* td)
{
  //  cerr << "Register: td=" << td << ", name=" << td->getName() << '\n';
  if(!types)
    types=scinew map<string, const TypeDescription*>;
  (*types)[td->getName()]=td;
}

TypeDescription::Register::~Register()
{
}

MPI_Datatype TypeDescription::getMPIType() const
{
   if(d_mpitype == -1){
      if(d_mpitypemaker){
	 d_mpitype = (*d_mpitypemaker)();
      } else {
	 throw InternalError("MPI Datatype requested, but do not know how to make it");
      }
   }
   ASSERT(d_mpitype != -2);
   return d_mpitype;
}

Variable* TypeDescription::createInstance() const
{
   if(!d_maker)
      throw InternalError("Do not know how to create instance for type: "+getName());
   return (*d_maker)();
}


//
// $Log$
// Revision 1.9  2001/01/08 22:12:12  jas
// Added switch for invalidFace for fillFlux, and friends.
// Added check for types in lookupType().
//
// Revision 1.8  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.7  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.6  2000/05/30 20:19:35  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.5  2000/05/21 08:19:09  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.4  2000/05/20 23:08:12  guilkey
// Fixed type database initialization.
//
// Revision 1.3  2000/05/20 08:09:28  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.2  2000/05/18 18:41:14  kuzimmer
// Added Particle to Basis enum, created Type enum with Scalar,Point,Vector,Tensor,& Other
//
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

