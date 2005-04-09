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
 * list.h: defines the List struct
 *
 *************************************************************************/

#ifndef _LIST_H_
#define _LIST_H_

/* Quarks header files */
#include "types.h"
#include "util.h"

struct LBuffer;

typedef struct Listnode {
    void *object;
    struct Listnode *next;
    struct LBuffer  *pool_ptr;
} Listnode;

typedef struct LBuffer {
    Spinlock inuse;
    Listnode *buf;
} LBuffer;


typedef struct List {
    Listnode *head, *tail;
    Listnode *current;
} List;

extern List *new_list();
extern int list_append(List *, void *);
extern void *list_rem_head(List *);
extern void list_rem_item(List *, void *);
extern int list_empty(List *);

/*
extern void *list_next(List *);
extern void *list_item(List *);
extern void list_initialize(List *);
*/

#endif  /* _LIST_H_ */
