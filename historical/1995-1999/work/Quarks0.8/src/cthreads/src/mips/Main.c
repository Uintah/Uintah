static float foo = 1.0;
extern void **cthread_init();

copy_stack( register void **oldsp, register void **newsp, int size)
{
  register int i;
  for(i=0; i<size; i++, oldsp++, newsp++)
    if (*oldsp >= (void*)oldsp && *oldsp < (void *)oldsp+ size*sizeof(void*))
      *newsp = ((void *)newsp + (*oldsp  - (void *)oldsp));
    else
      *newsp = *oldsp;
}




Main(int argc, char **argv, char **envp)
{
  register void  **newsp = cthread_init() - 64;
  register void **oldsp;
  static int argc_;
  static char **argv_;
  static char **envp_;
  
  argc_ = argc;
  argv_ = argv;
  envp_ = envp;

  asm( "move %0,$sp" : "=r" (oldsp) ); /* get old stack pointer */
  copy_stack( oldsp, newsp, 64);
  asm( "move $sp,%0" :: "r" (newsp) );  /* set new stackpointer */
  asm( "move $fp,%0" :: "r" (newsp) );  /* set new stackpointer */
  main(argc_, argv_, envp_);
  cthread_exit();
  /*NOT REACHED*/
}

