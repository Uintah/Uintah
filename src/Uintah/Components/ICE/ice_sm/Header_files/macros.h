#if N_DIMENSIONS == 3
    #define IF_3D(a) (a)
#else
    #define IF_3D(a) 0
#endif


/*__________________________________
*   MAX macros
*___________________________________*/
#define MAX(x,y) ((x)>(y)?(x):(y))

#define MAX_0_X(x) ((x)>(0.0)?(x):(0.0))
/*__________________________________
*   Ghost cell function
*   calculate the upper and lower looping
*   indicies of the outer ghostcell layer
*___________________________________*/
#define GC_LO(A) \
    (A - N_GHOSTCELLS)
    
#define GC_HI(A) \
    (A + N_GHOSTCELLS)
    
/*__________________________________
*   Shut up fullwarn compiler remarks
*___________________________________*/
#define QUITE_FULLWARN(arg) if(arg)

/*__________________________________
*   define functions for 
*   initialize_darray functions
*___________________________________*/
#define USR_FUNCTION(funct,x,y,z,const)         \
(                                               \
    (funct == 0)?  (const):                     \
    (                                           \
        (funct == 1)?  (const + x):             \
        (                                       \
            (funct == 2)?  (const + y):         \
            (                                   \
                (funct == 3)?  (const + z):     \
                (                               \
                    (funct == 4)? (12.0*( pow(x,2.0) - x) ): \
                    (                           \
                        (funct == 5)? (12.0*(pow(y,2.0) - y) ):0\
                    )                           \
                )                               \
            )                                   \
        )                                       \
    )                                           \
)
