/*              Quarks distributed shared memory system.
 * 
 * Copyright (c) 1995 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the Computer 
 * Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 * 	Utah $Hdr$
 *	Author: Dilip Khandekar, University of Utah CSL
 */
/**************************************************************************
 *
 * mem.c: SIGSEGV handler and other vm operations. 
 *
 *************************************************************************/

#include <signal.h>
#ifdef IRIX
#include <siginfo.h>
#include <ucontext.h>
#endif

#include "types.h"
#include "ptable.h"
#include "protocol.h"
#include "region.h"
#include "config.h"
#include "instruction.h"

extern Any dsm_thread(Any);
extern void Qk_shutdown(int);

void 
#if defined(BSD_HP300) || defined(BSD_HP800) || defined(HPUX)
segv_handler(int sig, int code, struct sigcontext *scp)
#endif 
#if defined(SUNOS) 
segv_handler(int sig, int code, struct sigcontext *scp, char *a)
#endif 
#ifdef IRIX
segv_handler(int sig, siginfo_t *sip, ucontext_t *uc)  
#endif
{
    Address lva_addr, gva_addr;
    char    *addr;
    Id      regionid;
    int     read_fault, write_fault;
    Ptentry *pte;
    unsigned long instruction;
#ifdef IRIX
    unsigned long pc, *pcptr;
    unsigned long *addr_ptr;
#endif

    sigsetmask(0);

#if defined(BSD_HP300) || (BSD_HP800)
    if (sig == SIGSEGV)
#ifdef BSD_HP300
	addr = (char *) code;
#endif
#ifdef BSD_HP800
        addr = (char *) *(unsigned long *) (scp->sc_sp+92);
#endif
    else if (sig == SIGBUS)
	addr = (char *) code;
    else
	PANIC("Cannot handle signals other than SEGV and BUS");
#endif

#if defined(SUNOS)
    addr = a;
#endif

#ifdef IRIX
    addr = (char *) sip->si_addr;
#endif

#ifdef HPUX
    addr = (char *) scp->sc_sl.sl_ss.ss_cr21;
#endif

    lva_addr = (Address) addr;
    
    regionid = laddr2region(lva_addr);
    
    if (regionid == INVALID_ID)
	PANIC("Genuine segmentation fault found");

    /* page align the address: */
    lva_addr = PAGE_ALIGN(lva_addr);

    gva_addr = lva2gva(lva_addr);

#if defined(BSD_HP300) || defined(BSD_HP800) || defined(HPUX)
/*    PANIC("Get the PC out of sigcontext"); */
    instruction = *(unsigned long *) 0;
#endif

#if defined(SUNOS)
    instruction = *(unsigned long *) scp->sc_pc;
#endif

#if defined(IRIX)
    pc = (unsigned long) uc->uc_mcontext.gregs[35];
    pcptr = (unsigned long *) pc;
    instruction = *pcptr; 
#endif

    pte = get_pte(regionid, lva_addr);
    write_fault = STORE_INSTRUCTION(instruction);
    read_fault  = LOAD_INSTRUCTION(instruction);

    ASSERT(! (write_fault && read_fault));
    ASSERT(write_fault || read_fault);

    if (write_fault && (pte->protocol == PROT_WRITE_INVALIDATE))
	pte->inval_pending = 1;

    if (read_fault)
    {
	lock_pte(pte);
	(*pte->read_fault_handler)(gva_addr, lva_addr, pte);
	unlock_pte(pte);
    }
    if (write_fault)
    {
	lock_pte(pte);
	(*pte->write_fault_handler)(gva_addr, lva_addr, pte);
	unlock_pte(pte);
    }
}


void 
#if defined(BSD_HP300) || defined(BSD_HP800) || defined(HPUX)
intr_handler(int sig, int code, struct sigcontext *scp)
#endif 
#if defined(SUNOS) 
intr_handler(int sig, int code, struct sigcontext *scp, char *a)
#endif 
#ifdef IRIX
intr_handler(int sig, siginfo_t *sip, ucontext_t *uc) 
#endif
{
    Qk_shutdown(1);
}

void quarks_dsm_init()
{
    int i;

#ifdef SUNOS
    signal(SIGSEGV, segv_handler);
#endif

#if defined(BSD_HP300) || defined(BSD_HP800)
    struct sigvec vec;
    vec.sv_handler = (SFPTR) segv_handler;
    vec.sv_mask    = 0;
    vec.sv_flags   = 0;

    sigvec(SIGSEGV, &vec, 0);
    sigvec(SIGBUS,  &vec, 0);
#endif

#ifdef IRIX
    struct sigaction act;
    act.sa_flags = SA_SIGINFO;

    act.sa_handler = (__sigret_t (*)(_sigargs)) segv_handler;
    for (i=0; i<4; i++) act.sa_mask.sigbits[i] = 0;
    if (sigaction(SIGSEGV, &act, 0) < 0)
	perror("SIGACTION-SIGSEGV");
    if (sigaction(SIGBUS,  &act, 0) < 0)
	perror("SIGACTION-SIGBUS");
#endif

#ifdef HPUX
    signal(SIGBUS, (void (*)(int)) segv_handler);
#endif

    DUQ = new_list();
    init_lltable();     /* local lock table */
    init_lbtable();     /* local barrier table */
    quarks_thread_fork(dsm_thread, 0);
}


void quarks_dsm_shutdown()
{
    close_all_regions();
    free_all_locks();
    free_all_barriers(); 
}
