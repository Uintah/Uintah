/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.

*/
	.option	pic2
	.text	
	.align	2
	.globl	Allocator_try_lock
	.ent	Allocator_try_lock 2
Allocator_try_lock:
	.frame	$sp, 0, $31
	.set noreorder
again:	
	ll	$2, 0($4)
	li	$5, 1
	sc	$5, 0($4)
	nop
	beqz	$5, again
        nop
	j	$31
	nop

	.end	Allocator_try_lock
