#ifndef UINTAH_HOMEBREW_PROCESSORCONTEXT_H
#define UINTAH_HOMEBREW_PROCESSORCONTEXT_H

namespace SCICore {
    namespace Thread {
	class SimpleReducer;
    }
}

namespace Uintah {

   using SCICore::Thread::SimpleReducer;

/**************************************

CLASS
   ProcessorContext
   
   Short description...

GENERAL INFORMATION

   ProcessorContext.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Processor_Context

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ProcessorContext {
   public:
      ~ProcessorContext();
      
      //////////
      // Insert Documentation Here:
      static ProcessorContext* getRootContext();
      
      //////////
      // Insert Documentation Here:
      int numThreads() const {
	 return d_numThreads;
      }
      
      //////////
      // Insert Documentation Here:
      int threadNumber() const {
	 return d_threadNumber;
      }
      
      //////////
      // Insert Documentation Here:
      void setNumThreads(int numThreads) {
	 d_numThreads = numThreads;
      }
      
      //////////
      // Insert Documentation Here:
      ProcessorContext* createContext(int threadNumber, 
				      int numThreads,
				      SimpleReducer* reducer) const;
      //////////
      // Insert Documentation Here:
      void barrier_wait() const;
      
      //////////
      // Insert Documentation Here:
      double reduce_min(double) const;
      
   private:
      //////////
      // Insert Documentation Here:
      const ProcessorContext*  d_parent;
      SimpleReducer*     d_reducer;
      int                d_numThreads;
      int                d_threadNumber;
      
      ProcessorContext(const ProcessorContext* parent,
		       int threadNumber, int numThreads,
		       SCICore::Thread::SimpleReducer* reducer);
      
      ProcessorContext(const ProcessorContext&);
      ProcessorContext& operator=(const ProcessorContext&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/04/26 06:49:15  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:08:39  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
