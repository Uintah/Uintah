	.verstamp	3 19
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
	li	$5, 10
	sc	$5, 0($4)
	nop
	beqz	$5, again
	j	$31

	.livereg	0x2000FF0E,0x00000FFF
	.end	Allocator_try_lock
