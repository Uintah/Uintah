	.option	pic2
	.text	
	.align	2
	.globl	Task_try_lock
	.ent	Task_try_lock 2
Task_try_lock:
	.frame	$sp, 0, $31
	.set noreorder
again:	
	ll	$2, 0($4)
	li	$5, 10
	sc	$5, 0($4)
	nop
	beqz	$5, again
	nop
	j	$31

	.livereg	0x2000FF0E,0x00000FFF
	.end	Task_try_lock
