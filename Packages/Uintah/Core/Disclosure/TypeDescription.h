#ifndef UINTAH_HOMEBREW_TypeDescription_H
#define UINTAH_HOMEBREW_TypeDescription_H

#include <string>
#include <mpi.h>
#include <Core/Thread/Mutex.h>

namespace Uintah {
using std::string;
using namespace SCIRun;

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
	 SFCXVariable,
	 SFCYVariable,
	 SFCZVariable,
	 ParticleVariable,
	 PerPatch,
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

      TypeDescription(Type type, const string& name,
		      bool isFlat, MPI_Datatype (*make_mpitype)());
      TypeDescription(Type type, const string& name,
		      bool isFlat, MPI_Datatype mpitype);
      TypeDescription(Type type, const string& name,
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
      string getName() const;

      bool isFlat() const {
	 return d_isFlat;
      }

      MPI_Datatype getMPIType() const;

      struct Register {
	 Register(const TypeDescription*);
	 ~Register();
      };
      static const TypeDescription* lookupType(const string&);

      Variable* createInstance() const;

      ~TypeDescription();

   private:
      Type d_type;
      const TypeDescription* d_subtype;
      string d_name;
      bool d_isFlat;
      mutable MPI_Datatype d_mpitype;
      MPI_Datatype (*d_mpitypemaker)();
      Variable* (*d_maker)();

      mutable Mutex d_lock;
       
      TypeDescription(const TypeDescription&);
      TypeDescription& operator=(const TypeDescription&);

     void register_type();
   };

} // End namespace Uintah


#endif

