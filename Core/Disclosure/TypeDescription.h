#ifndef UINTAH_HOMEBREW_TypeDescription_H
#define UINTAH_HOMEBREW_TypeDescription_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

namespace Uintah {
using std::string;

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
	 short_int_type,
         long_type,
         long64_type,
         Short27,   // for Fracture
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

     static void deleteAll();
     
   private:
      Type d_type;
      const TypeDescription* d_subtype;
      string d_name;
      bool d_isFlat;
      mutable MPI_Datatype d_mpitype;
      MPI_Datatype (*d_mpitypemaker)();
      Variable* (*d_maker)();

      TypeDescription(const TypeDescription&);
      TypeDescription& operator=(const TypeDescription&);

     void register_type();
   };

} // End namespace Uintah


#endif

