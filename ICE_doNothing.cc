#include <stdio.h>

#include <Packages/Uintah/CCA/Ports/CFDInterface.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Core/Geometry/Vector.h>

using namespace SCIRun;

namespace Uintah {

class ProcessorGroup;

class ICE : public CFDInterface {
public:
   ICE();
   virtual ~ICE();
};
} // end namespace Uintah

extern "C" void audit();
using Uintah;

/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::doNothing--
 Filename:  ICE_doNothing.cc
 Purpose:   This file is used to make a dummy ICE library that is used
            by sus.  To really compile all of ICE you need to set the 
            environmental variable ICE = yes.  This prevents other uses of sus
            from dealing with any bugs in the ICE code.

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     Todd Harman   06/28/00                              
_____________________________________________________________________*/
ICE::ICE()
{
fprintf(stderr, "\n\n\nIf you really want to compile all of ICE you must set \n");
fprintf(stderr, "the environmental variable ICE = yes.  Otherwise this is \n");
fprintf(stderr, " is what is in the ICE library.  This is tacky but it \n");
fprintf(stderr, "it compartmentalizes the make of sus\n");
getchar();
exit(1);
}

ICE::~ICE()
} // End namespace Uintah
{

