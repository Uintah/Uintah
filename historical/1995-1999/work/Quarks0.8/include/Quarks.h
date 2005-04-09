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
 * Quarks.h: included in application programs
 *
 *************************************************************************/

#ifndef _QUARKS_H_
#define _QUARKS_H_

#include "types.h"

#define Qk_initialize()  (Qk_init(argc, argv, "Qk.hosts"))

extern Id       Qknodeid;
extern int      Qknumnodes;
extern int      Qkmaster;
extern int      Qkchild;

extern void     Qk_init(int argc, char **argv, char *script_file);
extern void     Qk_shutdown(int terminate);

extern void	Qk_create_region(Id regionid, Size size, 
				 unsigned long *var_addr);
extern Address  Qk_open_region(Id regionid);
extern void     Qk_close_region(Id regionid);
extern void     Qk_destroy_region(Id regionid);

extern int      Qk_fork(void *func);
extern void     Qk_wait_for_threads();

extern void     Qk_acquire(Id lockid);
extern void     Qk_release(Id lockid);
extern Id       Qk_newlock();
extern Id       Qk_newbarrier();
extern void     Qk_wait_barrier(Id ba_id, int num_crossers);

extern unsigned long Qk_current_time();
                 
#endif  /* _QUARKS_H_ */
