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
 * sparc_insruction.h: sparc specific.
 *
 *************************************************************************/

#ifndef _SPARC_INSTRUCTION_H_
#define _SPARC_INSTRUCTION_H_

#ifndef sparc
THIS FILE IS ONLY FOR SPARC INSTRUCTION SET
#endif

#define op1_mask   0xC0000000
#define op2_mask   0x01F80000
#define op1_shift  24
#define op2_shift  19

#define OP_WIDTH 8
#define num_load_opcodes 20
static char *ld_op_bits[num_load_opcodes] = { 
    "11000000", /* LD        */
    "11000001", /* LDUB      */
    "11000011", /* LDD       */
    "11110011", /* LDDC      */
    "11100011", /* LDDF      */
    "11100000", /* LDF       */
    "11010000", /* LDA   (p) */
    "11110000", /* LDC       */
    "11110001", /* LDCSR     */
    "11010011", /* LDDA  (p) */
    "11100001", /* LDFSR     */
    "11001001", /* LDSB      */
    "11011001", /* LDSBA (p) */
    "11001010", /* LDSH      */
    "11011010", /* LDSHA (p) */
    "11001101", /* LDSTUB    */
    "11011101", /* LDSTUBA(p)*/
    "11010001", /* LDUBA (p) */
    "11000010", /* LDUH      */
    "11010010",	/* LDUHA (p) */
};

#define num_store_opcodes 16
static char *st_op_bits[num_store_opcodes] = {
    "11000100", /* ST        */
    "11010100", /* STA   (p) */
    "11000101", /* STB       */
    "11010101", /* STBA  (p) */
    "11110100", /* STC       */
    "11110101", /* STCR      */
    "11000111", /* STD       */
    "11010111", /* STDA  (p) */
    "11110111", /* STDC      */
    "11110110", /* STDCQ (p) */
    "11100111", /* STDF      */
    "11100110", /* STDFQ (p) */
    "11100100", /* STF       */
    "11100101", /* STFSR     */
    "10000110", /* STH       */
    "11010110", /* STHA  (p) */
};


#endif  /* _SPARC_INSTRUCTION_H_ */
