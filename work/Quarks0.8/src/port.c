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
 * port.c: implements the local and remote port abstraction of UDP socket
 *
 *************************************************************************/

#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#ifdef IRIX
#include <siginfo.h>
#include <ucontext.h>
#endif

/* Quarks header files */
#include "port.h"
#include "types.h"
#include "util.h"

#include "cthreads.h"

Port        *localport;
Id          local_hostid;   /* actually the IP address */
int         socknum;        /* local socket for connections */

#ifdef HPUX
cthread_t idle_thread_id;
#endif

extern void
#if defined(BSD_HP300) || defined(BSD_HP800) || defined(HPUX)
handle_asyncio(int sig, int code, struct sigcontext *scp);
#endif 
#if defined(SUNOS) 
handle_asyncio(int sig, int code, struct sigcontext *scp, char *a);
#endif 
#ifdef IRIX
handle_asyncio(int sig, siginfo_t *sip, ucontext_t *uc);
#endif

#ifdef HPUX
void idlethread()
{
    while (1)
    {
	handle_asyncio(0, 0, 0); 
	cthread_yield();
    }
    return;
}
#endif

static void async_init()
{
#ifdef IRIX
    signal(SIGIO,  handle_asyncio);
#else
    signal(SIGIO, handle_asyncio); 
#endif

#ifdef HPUX
    if (fcntl(socknum, F_SETFL, O_NDELAY) < 0)
    {
	perror("fcntl F_SETFL O_NDELAY");
	PANIC("Could not set up asyncio");
    }
    idle_thread_id = cthread_fork((cthread_fn_t) idlethread, 0);
#else
    if (fcntl(socknum, F_SETOWN, getpid()) < 0)
    {
	perror("fcntl F_SETOWN");
	PANIC("Could not set up asyncio");
    }

    if (fcntl(socknum, F_SETFL, FASYNC|FNDELAY) < 0)
    {
	perror("fcntl F_SETFL FASYNC|FNDELAY");
	PANIC("Could not set up asyncio");
    }
#endif

}

void init_localport(int portnum)
{
    /* Set up a socket to enable communication. In case of the 
     * server, bind it to the port "portnum". Find a free port
     * to bind to, in case of a client.
     */

    struct hostent *ent;
    char   hostname[256];

    localport = (Port *) malloc(sizeof(Port));

    localport->sin.sin_addr.s_addr = INADDR_ANY;

    if ((socknum = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
	perror("socket");
	PANIC("Cannot create local socket");
    }
    if (portnum > 0)
    {
	localport->sin.sin_port = (unsigned short) portnum;
	if (bind(socknum, (struct sockaddr *) &localport->sin, 
		 sizeof(localport->sin)) < 0)
	{
	    perror("bind");
	    PANIC("Cannot bind socket to the address");
	}
    }
    else
    {
	unsigned short p = SERVER_PORTNUM + 1;
	while (p <= 0xffff)
	{
	    localport->sin.sin_port = p;
	    if (bind(socknum, (struct sockaddr *) &localport->sin, 
		     sizeof(localport->sin)) < 0)
	    {
		if (errno != EADDRINUSE)
		{
		    perror("bind");
		    PANIC("Cannot bind socket to address");
		}
		p++;
	    }
	    else 
		break;
	}
	if (p >= 0xffff)
	    PANIC("No free port number");
    }

    localport->sin.sin_family = AF_INET;

    /* get my host name and obtain my internet address */
    if (gethostname(hostname, 256) < 0)
	perror("gethostname");

    if (!(ent = gethostbyname(hostname)))
    {
	perror("gethostbyname");
    }
    localport->sin.sin_addr.s_addr = *(unsigned long *) ent->h_addr_list[0];

    local_hostid = localport->sin.sin_addr.s_addr;
    async_init();
}

int get_packet(Port *from, char *buffer, int maxlen)
{
    /* returns numeber of bytes received. If no message, it returns
     * 0. In case of an error it returns -1.
     */

    int r;
    int s = sizeof(from->sin);
    
    if ((r = recvfrom(socknum, buffer, maxlen, 0,
		      (struct sockaddr *) &from->sin, &s)) < 0)
    {
	if (errno != EWOULDBLOCK)
	    perror("recvfrom");
	else
	    r = 0;
    }
    return r;
}

void send_packet(Port *to, char *message, int size)
{
    /* Sends the message "message" of size "size" to the port "to"
     */
    int bytes_sent;

    bytes_sent = sendto(socknum, message, size, 0, 
			(struct sockaddr *) &to->sin, sizeof(to->sin));
    if (bytes_sent != size)
    {
	perror("sendto");
	PANIC("Could not send the packet correctly");
    }
}

void copy_port(Port *to, Port *from)
{	
    /* bitwise copy of the port addresses */
    mem_copy((char *) &from->sin, (char *) &to->sin, 
	     sizeof(from->sin));
}

int cmp_port(Port *a, Port *b)
{
    if ((a->sin.sin_port == b->sin.sin_port) && 
	(a->sin.sin_addr.s_addr == b->sin.sin_addr.s_addr))
	return 1;
    return 0;
}

