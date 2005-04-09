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
 * diff.c: encoding and decoding of diffs
 *
 *************************************************************************/

#include "ptable.h"
#include "config.h"

void
encode_object(unsigned long *newp, unsigned long *old, 
	     unsigned long *dest, int *size)
{
    unsigned long *start = newp;
    unsigned long *end = newp + PAGE_SIZE/4;
    unsigned long *src;
    unsigned long header;
    unsigned long st;
    int cnt, bytes;
    int s=0;

    while (newp < end)
    {
	for ( ; (newp < end) && (*newp == *old); newp++, old++);
	if (newp == end) break;

	cnt = 0;
	for (src = newp;
	     (newp < end) && (*newp != *old); 
	     newp++,old++) cnt++;
	
	header = (((src - start) & 0xffff) << 16) | cnt;

	*dest++ = header; s++;
	mem_copy((char *) src, (char *) dest, cnt*4);	
	dest += cnt; s += cnt;
    }
    *size = s;  /* size is in words */
}

void
decode_object(unsigned long *msgbuf, Ptentry *pte, int size)
{
    unsigned long header;
    unsigned long offset, runlength;
    unsigned long st;
    int readonly;

    /* pte cannot be invalid. it can be readonly or read-writable */
    ASSERT(pte->access != PAGE_MODE_NO);
    readonly = (pte->access == PAGE_MODE_RO);
    
    set_access(pte, PAGE_MODE_RW);
    while (size > 0)
    {
	header = *msgbuf;
	msgbuf++; size--;
	runlength = header & 0x0000ffff;       /* in words */
	offset = (header & 0xffff0000) >> 16;  /* in words */

	if (runlength == 0) 
	{
	    printf("Runlength has become zero. But size is still %d\n",
		   size);
	    break;
	}
	mem_copy((char *) msgbuf, (char *) (pte->addr + offset*4),
		      runlength*4);
	if (pte->twin_present)
	    mem_copy((char *) msgbuf,
			  (char *) (pte->twin_addr + offset*4), runlength*4);

	size -= runlength;
	msgbuf += runlength;
	if (size < 0) 
	    PANIC("Something dramatically wrong in DecodeObject! ");
    }
    if (readonly)
	set_access(pte, PAGE_MODE_RO);
}
