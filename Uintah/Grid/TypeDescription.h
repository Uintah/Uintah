#ifndef UINTAH_HOMEBREW_TypeDescription_H
#define UINTAH_HOMEBREW_TypeDescription_H

#include <string>
#include <mpi.h>

namespace Uintah {
   class Variable;
   
   /**************************************
     
     CLASS
       TypeDescription
      
       Short Description...
      
     GENERAL INFORMATION
      
       TypeDescription.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       TypeDescription
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class TypeDescription {
   public:
      enum Type {
	 CCVariable,
	 NCVariable,
	 XFCVariable,
	 YFCVariable,
	 ZFCVariable,
	 SFCXVariable,
	 SFCYVariable,
	 SFCZVariable,
	 ParticleVariable,
	 ScatterGatherVariable,
	 Point,
	 Vector,
	 Matrix3,
	 ReductionVariable,
	 double_type,
	 bool_type,
	 int_type,
         long_type,
	 Unknown,
	 Other
      };

      TypeDescription(Type type, const std::string& name,
		      bool isFlat, MPI_Datatype (*make_mpitype)());
      TypeDescription(Type type, const std::string& name,
		      bool isFlat, MPI_Datatype mpitype);
      TypeDescription(Type type, const std::string& name,
		      Variable* (*maker)(),
		      const TypeDescription* subtype);
     
      bool isReductionVariable() const {
	 return d_type == ReductionVariable;
      }
      Type getType() const {
	 return d_type;
      }
      const TypeDescription* getSubType() const {
	 return d_subtype;
      }
      std::string getName() const;

      bool isFlat() const {
	 return d_isFlat;
      }

      MPI_Datatype getMPIType() const;

      struct Register {
	 Register(const TypeDescription*);
	 ~Register();
      };
      static const TypeDescription* lookupType(const std::string&);

      Variable* createInstance() const;

      ~TypeDescription();

   private:
      Type d_type;
      const TypeDescription* d_subtype;
      std::string d_name;
      bool d_isFlat;
      mutable MPI_Datatype d_mpitype;
      MPI_Datatype (*d_mpitypemaker)();
      Variable* (*d_maker)();
       
      TypeDescription(const TypeDescription&);
      TypeDescription& operator=(const TypeDescription&);
       
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.10  2000/11/28 03:47:26  jas
// Added FCVariables for the specific faces X,Y,and Z.
//
// Revision 1.9  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.8  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.7  2000/06/27 23:18:18  rawat
// implemented Staggered cell variables. Modified Patch.cc to get ghostcell
// and staggered cell indexes.
//
// Revision 1.6  2000/06/14 23:27:34  jas
// Added FCVariables to enum.
//
// Revision 1.5  2000/06/02 17:22:14  guilkey
// Added long_type to the the TypeDescription and TypeUtils.
//
// Revision 1.4  2000/05/21 08:19:09  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.3  2000/05/20 08:09:29  sparker
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

#endif

