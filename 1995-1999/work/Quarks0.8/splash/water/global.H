#define MAXMOLS	1728

#include "Quarks.h"

 struct GlobalMemory {
        LOCKDEC(IntrafVirLock)
        LOCKDEC(InterfVirLock)
        LOCKDEC(FXLock)
        LOCKDEC(FYLock)
        LOCKDEC(FZLock)
	LOCKDEC(KinetiSumLock)
	LOCKDEC(PotengSumLock)
	ALOCKDEC(MolLock, MAXMOLS)
        BARDEC(start)
	BARDEC(InterfBar)
	BARDEC(PotengBar)
	double VIR;
	double SUM[3];
	double POTA, POTR, POTRF;
        volatile int initialized;    /* used to make sure gl->VIR is         */
                            /* initialized to 0 by process 0 before */
                            /* anybody accumulates into it in INTRAF*/
                            /* Used as a flag synch variable        */ 
        };

extern struct GlobalMemory *gl;

