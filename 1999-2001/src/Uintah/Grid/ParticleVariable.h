#ifndef UINTAH_HOMEBREW_PARTICLEVARIABLE_H
#define UINTAH_HOMEBREW_PARTICLEVARIABLE_H

#include <SCICore/Util/FancyAssert.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Util/Assert.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <Uintah/Grid/ParticleVariableBase.h>
#include <Uintah/Interface/InputContext.h>
#include <Uintah/Interface/OutputContext.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/ParticleData.h>
#include <Uintah/Grid/ParticleSubset.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Grid/TypeUtils.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <SCICore/Malloc/Allocator.h>
#include <unistd.h>
#include <errno.h>

namespace Uintah {
   using SCICore::Exceptions::ErrnoException;
   using SCICore::Exceptions::InternalError;
   using namespace PSECore::XMLUtil;
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
      ParticleVariable(ParticleData<T>*, ParticleSubset* pset);
      ParticleVariable(const ParticleVariable<T>&);
      
      ParticleVariable<T>& operator=(const ParticleVariable<T>&);

      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();

      //////////
      // Insert Documentation Here:
      void resync() {
	 d_pdata->resize(getParticleSet()->numParticles());
      }
      
      //////////
      // Insert Documentation Here:
      virtual ParticleVariableBase* clone() const;
      virtual ParticleVariableBase* cloneSubset(ParticleSubset*) const;
      
      //////////
      // Insert Documentation Here:
      inline T& operator[](particleIndex idx) {
	 ASSERTRANGE(idx, 0, (particleIndex)d_pdata->data.size());
	 return d_pdata->data[idx];
      }
      
      //////////
      // Insert Documentation Here:
      inline const T& operator[](particleIndex idx) const {
	 ASSERTRANGE(idx, 0, (particleIndex)d_pdata->data.size());
	 return d_pdata->data[idx];
      }
      
      virtual void copyPointer(const ParticleVariableBase&);
      virtual void allocate(ParticleSubset*);
      virtual void allocate(const Patch*)
      { throw InternalError("Should not call ParticleVariable<T>::allocate(const Patch*), use allocate(ParticleSubset*) instead."); }
      
      virtual void gather(ParticleSubset* dest,
			  std::vector<ParticleSubset*> subsets,
			  std::vector<ParticleVariableBase*> srcs,
			  particleIndex extra = 0);
      virtual void unpackMPI(void* buf, int bufsize, int* bufpos,
			     const ProcessorGroup* pg, int start, int n);
      virtual void packMPI(void* buf, int bufsize, int* bufpos,
			   const ProcessorGroup* pg, int start, int n);
      virtual void packsizeMPI(int* bufpos,
			       const ProcessorGroup* pg, int start, int n);
      virtual void emit(OutputContext&);
      virtual void read(InputContext&);
      virtual void* getBasePointer();
      virtual const TypeDescription* virtualGetTypeDescription() const;
   private:
      
      //////////
      // Insert Documentation Here:
      ParticleData<T>* d_pdata;
      
      static TypeDescription::Register registerMe;
      static Variable* maker();
   };

   template<class T>
      TypeDescription::Register ParticleVariable<T>::registerMe(getTypeDescription());

   template<class T>
      const TypeDescription*
      ParticleVariable<T>::getTypeDescription()
      {
	 static TypeDescription* td;
	 if(!td){
	    td = scinew TypeDescription(TypeDescription::ParticleVariable,
					"ParticleVariable", &maker,
					fun_getTypeDescription((T*)0));
	 }
	 return td;
      }
   
   template<class T>
      Variable*
      ParticleVariable<T>::maker()
      {
	 return scinew ParticleVariable<T>();
      }
   
   template<class T>
      ParticleVariable<T>::ParticleVariable()
      : ParticleVariableBase(0), d_pdata(0)
      {
      }
   
   template<class T>
      ParticleVariable<T>::~ParticleVariable()
      {
	 if(d_pdata && d_pdata->removeReference())
	    delete d_pdata;
      }
   
   template<class T>
      ParticleVariable<T>::ParticleVariable(ParticleSubset* pset)
      : ParticleVariableBase(pset)
      {
	 d_pdata=scinew ParticleData<T>(pset->getParticleSet()->numParticles());
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
	 d_pdata=scinew ParticleData<T>(pset->getParticleSet()->numParticles());
	 d_pdata->addReference();
      }
   
   template<class T>
      ParticleVariableBase*
      ParticleVariable<T>::clone() const
      {
	 return scinew ParticleVariable<T>(*this);
      }
   
   template<class T>
      ParticleVariableBase*
      ParticleVariable<T>::cloneSubset(ParticleSubset* pset) const
      {
	 return scinew ParticleVariable<T>(d_pdata, pset);
      }

   template<class T>
      ParticleVariable<T>::ParticleVariable(ParticleData<T>* pdata,
					    ParticleSubset* pset)
      : ParticleVariableBase(pset), d_pdata(pdata)
      {
	 if(d_pdata)
	    d_pdata->addReference();
      }
   
   template<class T>
      ParticleVariable<T>::ParticleVariable(const ParticleVariable<T>& copy)
      : ParticleVariableBase(copy), d_pdata(copy.d_pdata)
      {
	 if(d_pdata)
	    d_pdata->addReference();
      }
   
   template<class T>
      ParticleVariable<T>&
      ParticleVariable<T>::operator=(const ParticleVariable<T>& copy)
      {
	 if(this != &copy){
	    ParticleVariableBase::operator=(copy);
	    if(d_pdata && d_pdata->removeReference())
	       delete d_pdata;
	    d_pdata = copy.d_pdata;
	    if(d_pdata)
	       d_pdata->addReference();
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
				  std::vector<ParticleVariableBase*> srcs,
				  particleIndex extra)
      {
	 if(d_pdata && d_pdata->removeReference())
	    delete d_pdata;
	 if(d_pset && d_pset->removeReference())
	    delete d_pset;
	 d_pset = pset;
	 pset->addReference();
	 d_pdata=scinew ParticleData<T>(pset->getParticleSet()->numParticles());
	 d_pdata->addReference();
	 ASSERTEQ(subsets.size(), srcs.size());
	 ParticleSubset::iterator dstiter = pset->begin();
	 for(int i=0;i<(int)subsets.size();i++){
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
	 ASSERT(dstiter+extra == pset->end());
      }

   template<class T>
      void
      ParticleVariable<T>::emit(OutputContext& oc)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 appendElement(oc.varnode, "numParticles", d_pset->numParticles());
	 if(td->isFlat()){
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
	       ssize_t size = (ssize_t)(sizeof(T)*(end-start));
	       ssize_t s=write(oc.fd, &(*this)[start], size);
	       if(size != s)
		  throw ErrnoException("ParticleVariable::emit (write call)", errno);
	       oc.cur+=size;
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      void
      ParticleVariable<T>::read(InputContext& ic)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 if(td->isFlat()){
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
	       ssize_t size = (ssize_t)(sizeof(T)*(end-start));
	       ssize_t s=::read(ic.fd, &(*this)[start], size);
	       if(size != s)
		  throw ErrnoException("ParticleVariable::emit (write call)", errno);
	       ic.cur+=size;
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      void*
      ParticleVariable<T>::getBasePointer()
      {
	 return &d_pdata->data[0];
      }

   template<class T>
      const TypeDescription*
      ParticleVariable<T>::virtualGetTypeDescription() const
      {
	 return getTypeDescription();
      }
   
   template<class T>
      void
      ParticleVariable<T>::unpackMPI(void* buf, int bufsize, int* bufpos,
				     const ProcessorGroup* pg,
				     int start, int n)
      {
	 // This should be fixed for variable sized types!
	 const TypeDescription* td = getTypeDescription()->getSubType();
	 if(td->isFlat()){
	    ParticleSubset::iterator beg = d_pset->seek(start);
	    ParticleSubset::iterator end = d_pset->seek(start+n);
	    for(ParticleSubset::iterator iter = beg; iter != end; iter++){
	       MPI_Unpack(buf, bufsize, bufpos,
			  &d_pdata->data[*iter], 1, td->getMPIType(),
			  pg->getComm());
	    }
	 } else {
	    throw InternalError("packMPI not finished\n");
	 }
      }

   template<class T>
      void
      ParticleVariable<T>::packMPI(void* buf, int bufsize, int* bufpos,
			   const ProcessorGroup* pg, particleIndex start, int n)
      {
	 // This should be fixed for variable sized types!
	 const TypeDescription* td = getTypeDescription()->getSubType();
	 if(td->isFlat()){
	    ParticleSubset::iterator beg = d_pset->seek(start);
	    ParticleSubset::iterator end = d_pset->seek(start+n);
	    for(ParticleSubset::iterator iter = beg; iter != end; iter++){
	       MPI_Pack(&d_pdata->data[*iter], 1, td->getMPIType(),
			buf, bufsize, bufpos, pg->getComm());
	    }
	 } else {
	    throw InternalError("packMPI not finished\n");
	 }
      }

   template<class T>
      void
      ParticleVariable<T>::packsizeMPI(int* bufpos,
			       const ProcessorGroup* pg, int, int n)
      {
	 // This should be fixed for variable sized types!
	 const TypeDescription* td = getTypeDescription()->getSubType();
	 if(td->isFlat()){
	    int size;
	    MPI_Pack_size(n, td->getMPIType(), pg->getComm(), &size);
	    (*bufpos)+= size;
	 } else {
	    throw InternalError("packsizeMPI not finished\n");
	 }
      }

} // end namespace Uintah

//
// $Log$
// Revision 1.23  2000/12/23 00:32:47  witzel
// Added emit(OutputContext), read(InputContext), and allocate(Patch*) as
// pure virtual methods to class Variable and did any needed implementations
// of these in sub-classes.
//
// Revision 1.22  2000/11/28 01:15:48  guilkey
// Inlined the [] operator.
//
// Revision 1.21  2000/09/25 20:37:42  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.20  2000/09/25 18:12:20  sparker
// do not use covariant return types due to problems with g++
// other linux/g++ fixes
//
// Revision 1.19  2000/08/08 01:32:46  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.18  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.17  2000/06/15 21:57:18  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.16  2000/06/05 20:56:42  tan
// Added const T& operator[](particleIndex idx) const.
//
// Revision 1.15  2000/05/30 20:19:31  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.14  2000/05/20 08:09:25  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
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
// Made patches have a single uniform index space - still needs work
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
