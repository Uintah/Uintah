#ifndef UINTAH_HOMEBREW_PARTICLEVARIABLE_H
#define UINTAH_HOMEBREW_PARTICLEVARIABLE_H

#include <SCICore/Util/FancyAssert.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Exceptions/InternalError.h>
#include <Uintah/Grid/EmitUtils.h>
#include <Uintah/Grid/ParticleVariableBase.h>
#include <Uintah/Interface/OutputContext.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/ParticleData.h>
#include <Uintah/Grid/ParticleSubset.h>
#include <Uintah/Grid/TypeDescription.h>
#include <unistd.h>
#include <errno.h>

namespace Uintah {
   using SCICore::Exceptions::ErrnoException;
   using SCICore::Exceptions::InternalError;
   class TypeDescription;

/**************************************

CLASS
   ParticleVariable
   
   Short description...

GENERAL INFORMATION

   ParticleVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Particle_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   template<class T> class ParticleVariable : public ParticleVariableBase {
   public:
      ParticleVariable();
      virtual ~ParticleVariable();
      ParticleVariable(ParticleSubset* pset);
      ParticleVariable(const ParticleVariable<T>&);
      
      ParticleVariable<T>& operator=(const ParticleVariable<T>&);

      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();
      
      //////////
      // Insert Documentation Here:
      ParticleSet* getParticleSet() const {
	 return d_pset->getParticleSet();
      }
      
      //////////
      // Insert Documentation Here:
      ParticleSubset* getParticleSubset() const {
	 return d_pset;
      }
      //////////
      // Insert Documentation Here:
     void resync() {
       d_pdata->resize(getParticleSet()->numParticles());
      }
      
      //////////
      // Insert Documentation Here:
      virtual ParticleVariable<T>* clone() const;
      
      //////////
      // Insert Documentation Here:
      T& operator[](particleIndex idx) {
	 ASSERTRANGE(idx, 0, d_pdata->data.size());
	 return d_pdata->data[idx];
      }
      
      virtual void copyPointer(const ParticleVariableBase&);
      virtual void allocate(ParticleSubset*);
      virtual void gather(ParticleSubset* dest,
			  std::vector<ParticleSubset*> subsets,
			  std::vector<ParticleVariableBase*> srcs);
      virtual void emit(OutputContext&);
   private:
      
      //////////
      // Insert Documentation Here:
      ParticleData<T>* d_pdata;
      ParticleSubset*  d_pset;
      
   };
   
   template<class T>
      const TypeDescription*
      ParticleVariable<T>::getTypeDescription()
      {
	 static TypeDescription* td;
	 if(!td)
	    td = new TypeDescription(false, TypeDescription::Cell);
	 return td;
      }
   
   template<class T>
      ParticleVariable<T>::ParticleVariable()
      : d_pdata(0), d_pset(0)
      {
      }
   
   template<class T>
      ParticleVariable<T>::~ParticleVariable()
      {
	 if(d_pdata && d_pdata->removeReference())
	    delete d_pdata;
	 if(d_pset && d_pset->removeReference())
	    delete d_pset;
      }
   
   template<class T>
      ParticleVariable<T>::ParticleVariable(ParticleSubset* pset)
      : d_pset(pset)
      {
	 d_pset->addReference();
	 d_pdata=new ParticleData<T>(pset->getParticleSet()->numParticles());
	 d_pdata->addReference();
      }
   
   template<class T>
      void ParticleVariable<T>::allocate(ParticleSubset* pset)
      {
	 if(d_pdata && d_pdata->removeReference())
	    delete d_pdata;
	 if(d_pset && d_pset->removeReference())
	    delete d_pset;

	 d_pset=pset;
	 d_pset->addReference();
	 d_pdata=new ParticleData<T>(pset->getParticleSet()->numParticles());
	 d_pdata->addReference();
      }
   
   template<class T>
      ParticleVariable<T>*
      ParticleVariable<T>::clone() const
      {
	 return new ParticleVariable<T>(*this);
      }
   
   template<class T>
      ParticleVariable<T>::ParticleVariable(const ParticleVariable<T>& copy)
      : d_pdata(copy.d_pdata), d_pset(copy.d_pset)
      {
	 if(d_pdata)
	    d_pdata->addReference();
	 if(d_pset)
	    d_pset->addReference();
      }
   
   template<class T>
      ParticleVariable<T>&
      ParticleVariable<T>::operator=(const ParticleVariable<T>& copy)
      {
	 if(this != &copy){
	    if(d_pdata && d_pdata->removeReference())
	       delete d_pdata;
	    if(d_pset && d_pset->removeReference())
	       delete d_pset;
	    d_pset = copy.d_pset;
	    d_pdata = copy.d_pdata;
	    if(d_pdata)
	       d_pdata->addReference();
	    if(d_pset)
	       d_pset->addReference();
	 }
	 return *this;
      }
   
   template<class T>
      void
      ParticleVariable<T>::copyPointer(const ParticleVariableBase& copy)
      {
	 const ParticleVariable<T>* c = dynamic_cast<const ParticleVariable<T>* >(&copy);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in particle variable");
	 *this = *c;
      }

   template<class T>
      void
      ParticleVariable<T>::gather(ParticleSubset* pset,
				  std::vector<ParticleSubset*> subsets,
				  std::vector<ParticleVariableBase*> srcs)
      {
	 if(d_pdata && d_pdata->removeReference())
	    delete d_pdata;
	 if(d_pset && d_pset->removeReference())
	    delete d_pset;
	 d_pset = pset;
	 pset->addReference();
	 d_pdata=new ParticleData<T>(pset->getParticleSet()->numParticles());
	 d_pdata->addReference();
	 ASSERTEQ(subsets.size(), srcs.size());
	 ParticleSubset::iterator dstiter = pset->begin();
	 for(int i=0;i<subsets.size();i++){
	    ParticleVariable<T>* srcptr = dynamic_cast<ParticleVariable<T>*>(srcs[i]);
	    if(!srcptr)
	       throw TypeMismatchException("Type mismatch in ParticleVariable::gather");
	    ParticleVariable<T>& src = *srcptr;
	    ParticleSubset* subset = subsets[i];
	    for(ParticleSubset::iterator srciter = subset->begin();
		srciter != subset->end(); srciter++){
	       (*this)[*dstiter] = src[*srciter];
	       dstiter++;
	    }
	 }
	 ASSERT(dstiter == pset->end());
      }

   template<class T>
      void
      ParticleVariable<T>::emit(OutputContext& oc)
      {
	 T* t=0;
	 if(isFlat(*t)){
	    // This could be optimized...
	    ParticleSubset::iterator iter = d_pset->begin();
	    while(iter != d_pset->end()){
	       particleIndex start = *iter;
	       iter++;
	       particleIndex end = start+1;
	       while(iter != d_pset->end() && *iter == end) {
		  end++;
		  iter++;
	       }
	       size_t size = sizeof(T)*(end-start);
	       ssize_t s=write(oc.fd, &(*this)[start], size);
	       if(size != s)
		  throw ErrnoException("ParticleVariable::emit (write call)", errno);
	       oc.cur+=size;
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }
} // end namespace Uintah

//
// $Log$
// Revision 1.13  2000/05/20 02:36:06  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.12  2000/05/15 19:39:48  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.11  2000/05/10 20:03:01  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.10  2000/05/07 06:02:12  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.9  2000/05/01 16:18:17  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.8  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.7  2000/04/28 07:35:37  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.6  2000/04/26 06:48:52  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/20 18:56:30  sparker
// Updates to MPM
//
// Revision 1.4  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.3  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
