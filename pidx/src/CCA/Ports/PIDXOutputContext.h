#ifndef UINTAH_HOMEBREW_PIDXOutputContext_H
#define UINTAH_HOMEBREW_PIDXOutputContext_H

#include <sci_defs/pidx_defs.h>
#if HAVE_PIDX

#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <PIDX.h>
#include <string>

namespace Uintah {
   /**************************************
     
     CLASS
       PIDXOutputContext
      
       Short Description...
      
     GENERAL INFORMATION
      
       PIDXOutputContext.h
      
       Sidharth Kumar
       School of Computing
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       PIDXOutputContext
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
class PIDXOutputContext {
public:
  PIDXOutputContext(std::string filename, unsigned int timeStep,
                    int globalExtent[5] ,MPI_Comm comm);
  ~PIDXOutputContext();

  std::string filename;
  unsigned int timestep;
  int total_dimension;
  int fd;
  int global_dimension[5];
  MPI_Comm comm;
  
  /* added from test_pidx_writer.c */
    PIDX_file idx;
    PIDX_variable variable;

  //   private:
  //       OutputContext(const OutputContext&);
  //       OutputContext& operator=(const OutputContext&);
      
   };
} // End namespace Uintah

#endif //HAVE_PIDX
#endif //UINTAH_HOMEBREW_PIDXOutputContext_H
