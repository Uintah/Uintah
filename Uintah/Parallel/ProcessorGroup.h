#ifndef UINTAH_HOMEBREW_PROCESSORGROUP_H
#define UINTAH_HOMEBREW_PROCESSORGROUP_H

namespace Uintah {

/**************************************

CLASS
   ProcessorGroup
   
   Short description...

GENERAL INFORMATION

   ProcessorGroup.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Processor_Group

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ProcessorGroup {
   public:
      ~ProcessorGroup();
      
      //////////
      // Insert Documentation Here:
      int size() const {
	 return d_size;
      }

      //////////
      // Insert Documentation Here:
      int myrank() const {
	 return d_rank;
      }

   private:
      //////////
      // Insert Documentation Here:
      const ProcessorGroup*  d_parent;
      
      friend class Parallel;
      ProcessorGroup(const ProcessorGroup* parent,
		     int rank, int size);

      int d_rank;
      int d_size;
      
      ProcessorGroup(const ProcessorGroup&);
      ProcessorGroup& operator=(const ProcessorGroup&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/17 07:06:49  sparker
// Changed ProcessorContext to ProcessorGroup
//
//

#endif
