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
 * util.c: utilities
 *
 *************************************************************************/

#include <signal.h>
#include <sys/time.h>

/* Quarks header files */
#include "port.h"
#include "thread.h"
#include "util.h"

/* external dependencies */
extern Any  dsm_thread(Any);
extern void quarks_dsm_init();
char *Qksrvhname = 0;

static struct timeval tp;
static unsigned long start_sec  = 0; /* time of first invocation of */
static unsigned long start_msec = 0; /* Qk_current_time()              */

static int iloopcnt;

void mumble(char *message)
{
    fprintf(stdout, "%s\n", message);
}

void mem_zero(char *addr, int size)
{
    bzero(addr, size);
}

void mem_copy(char *from, char *to, int size)
{
#ifdef HPUX
    memmove((void *) to, (void *) from, size);
#else
    bcopy(from, to, size);
#endif
}

unsigned long Qk_current_time()
{
    gettimeofday(&tp, 0);
    if (!start_sec)
    {
	start_sec = tp.tv_sec;
	start_msec = (unsigned long) (tp.tv_usec/1000);
    }
    return (1000*(tp.tv_sec-start_sec) + (tp.tv_usec/1000 - start_msec));
}

void disable_signals()
{
    /* 5.72 usec on HP 700 */
    sigblock((int)(sigmask(SIGIO)|sigmask(SIGVTALRM)|sigmask(SIGIO)));
}

void enable_signals()
{
    /* 5.69 usec on HP 700 */
    sigsetmask(0);
}

void iloopck_init()
{
    iloopcnt=0;
}

void iloopck()
{
    iloopcnt++;
    if (iloopcnt > ILOOP_MAX)
	PANIC("Possible infinite loop");
}

void graceful_exit()
{
#ifdef SHM
    int shmid;

    for (shmid=0; shmid < MAXSHMID; shmid++)
	shmctl(shmid, IPC_RMID, 0);
#endif

    exit(1);
}    



void quarks_basic_init(int server)
{
    char server_hostname[256];

    if (server)
	init_localport(SERVER_PORTNUM);
    else
	init_localport(0);
    
    init_thread_tables();  /* also makes main() a thread */

    if (!server) 
    {
	bind_server(Qksrvhname);
	quarks_dsm_init();
	register_server();
	fprintf(stdout, "Quarks initialization done \n");
    }
    else
    {
	Qknodeid = 0;
	fprintf(stdout, "Server ready\n");
    }
}

