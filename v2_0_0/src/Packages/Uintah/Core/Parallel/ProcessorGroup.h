#ifndef UINTAH_HOMEBREW_PROCESSORGROUP_H
#define UINTAH_HOMEBREW_PROCESSORGROUP_H

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

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

      MPI_Comm getComm() const {
	 return d_comm;
      }

   private:
      //////////
      // Insert Documentation Here:
      const ProcessorGroup*  d_parent;
      
      friend class Parallel;
      ProcessorGroup(const ProcessorGroup* parent,
		     MPI_Comm comm, bool allmpi,
		     int rank, int size);

      int d_rank;
      int d_size;
      MPI_Comm d_comm;
      bool d_allmpi;
      
      ProcessorGroup(const ProcessorGroup&);
      ProcessorGroup& operator=(const ProcessorGroup&);
   };
} // End namespace Uintah
   


#endif
