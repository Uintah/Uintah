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
	li	$5, 1
	sc	$5, 0($4)
	nop
	beqz	$5, again
        nop
	j	$31

	.end	Allocator_try_lock
