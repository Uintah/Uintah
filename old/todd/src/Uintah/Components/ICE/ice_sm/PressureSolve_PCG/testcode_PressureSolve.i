
#if switchInclude_stencil_test_code
 k = 0;
  for ( j=0; j<n; j++ ) {
    for ( i=0; i<m; i++ ) {
      ae = -hy/hx;         /* east   */
      aw = -hy/hx;         /* west   */
      an = -hx/hy;         /* north  */
      as = -hx/hy;         /* south  */
      ap = -(ae+aw+an+as); /* center; conservative! */
      
      /*__________________________________
      * Tweek the boundaries
      *___________________________________*/
      if ( i == 0 ) {
         aw = hy;
         ap = -(ae+an+as);
      } else if ( i == m-1 ) {
         ae = hy;
         ap = -(aw+an+as);
      }  
      if ( j == 0 ) {
         as = hx;
         ap = -(ae+aw+an);
      } else if ( j == n-1 ) {
         an = hx;
         ap = -(ae+aw+as);
      }
      if ( i == 0 && j == 0 ) {
         ap = -(ae+an);
      } else if ( i == 0 && j == n-1 ) {
         ap = -(ae+as);
      } else  if ( i == m-1 && j == 0 ) {
         ap = -(aw+an);
      } else if ( i == m-1 && j == n-1 ) {
         ap = -(aw+as);
      }
      /*__________________________________
      * Finally set the stencil
      *___________________________________*/
      VecSetValue( stencil->ap, k, ap, INSERT_VALUES ); 
      VecSetValue( stencil->ae, k, ae, INSERT_VALUES ); 
      VecSetValue( stencil->aw, k, aw, INSERT_VALUES ); 
      VecSetValue( stencil->an, k, an, INSERT_VALUES ); 
      VecSetValue( stencil->as, k, as, INSERT_VALUES ); 
      k++;
    }
  }
#endif



/*______________________________________________________________________
*   Test code used by computeSource.c
*_______________________________________________________________________*/
#if switchInclude_source_test_code
  N     = m*n;
  ierr  = VecCreateSeq(PETSC_COMM_SELF, N, solution);                                       CHKERRQ(ierr);
  hx    = 1.0/m; 
  hy    = 1.0/n;
  y     = hy*0.5;
  k     = 0;
  for ( j=0; j<n; j++ ) 
  {
    x = hx*0.5;
    for ( i=0; i<m; i++ ) 
    {
      v = (4.0*pow(x,3.0) - 6.0*pow(x,2.0) + 1.0)*(4.0*pow(y,3.0) - 6.0*pow(y,2.0) + 1.0);
      VecSetValue( *solution, k, v, INSERT_VALUES );
      
      v = -12.0*hx*hy*((2.0*x - 1.0)*(4.0*pow(y,3.0) - 6.0*pow(y,2.0) + 1.0) +
                       (2.0*y - 1.0)*(4.0*pow(x,3.0) - 6.0*pow(x,2.0) + 1.0));
      VecSetValue( userctx->b, k, v, INSERT_VALUES );
      x += hx;
      k++;
    }
    y += hy;
  }
#endif


/*______________________________________________________________________
*   code used to compute the error
*_______________________________________________________________________*/
#if switchInclude_compute_error
  ierr = VecCreateSeq(PETSC_COMM_SELF, N, &error);                                          CHKERRQ(ierr);
  ierr = VecWAXPY( &MINUSONE, userctx.x, solution, error );                                 CHKERRQ(ierr);
  ierr = VecNorm(error, NORM_INFINITY, &enorm );

  printf("m %d n %d iterations %d error norm %g\n", m, n, its, enorm);
#endif
