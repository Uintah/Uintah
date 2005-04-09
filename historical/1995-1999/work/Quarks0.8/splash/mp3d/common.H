/* Space constants - maximum dimensions and such */

#include <math.h>
#define quarks

#ifdef quarks
#include "Quarks.h"
#endif

#define MAX_XCELL 24
#define MAX_ZCELL 8

/* Maximum molecule constants */

#ifdef MULTIMAX
#  define MAX_MOL 4000
#else
#  define MAX_MOL 50000
#endif

#ifdef quarks
#define MAX_MOL 4000
#endif

#define RES_RATIO 50

/* Mix table constants */

#define MIXS_POSSIBLE  120
#define SIGNS_POSSIBLE  32
#define MIX_DOF          5

/* Boundry condition constants */

#define COND_OFFSET 30000

/* Boolean definitions */

#ifndef TRUE
#  define TRUE 1
#endif
#ifndef FALSE
#  define FALSE 0
#endif

/* Buffer Length Definitions */

#define FILENAME_LENGTH 60

/* Odds and ends */

#define DF 5         /* Molecule degrees of freedom */
#define I_MAX 500    /* For temp plot space */
#define ALIGNMENT 64 

#ifdef MULTIMAX
#  define   MAX_PROCESSORS  12
#else
#  ifdef AUG
#    define MAX_PROCESSORS 256
#  else
#    define MAX_PROCESSORS   1
#  endif
#endif

#ifdef quarks
#define MAX_PROCESSORS 16
#endif

typedef LOCKDEC(alock)

struct particle {
  float x, 
        y, 
        z, 
        r, 
        s, 
        u, 
        v, 
        w;
  struct acell *where;
};

typedef struct acell {
#ifdef LOCKING
  alock space_lock;
#endif
  int space,
      cell_pop, 
      cum_cell_pop;
  float avg_prob, 
        U, 
        V, 
        W, 
        Usqr, 
        Vsqr, 
        Wsqr;
#ifdef LOCKING
  alock sample_lock;
#endif
} column[MAX_XCELL];  /* a column of cells in the x-direction */

typedef column plane[MAX_ZCELL];  /* a plane of cells in the xz-plane */

/* 
 * Define the Globally-shared data structures. 
 */

/* Particle Properties - positions and velocites */
MAIN struct particle *Particles;

/* Instant and cumulative cell populations and physical space map. */
MAIN plane *Cells;

/* Thermalizer reservoir variables */
MAIN struct ares {
  float xr, 
        yr, 
        zr, 
        rr, 
        sr, 
        ur, 
        vr, 
        wr;
} *Ares;

/* Boundary Conditions Temporary Space */
MAIN int *Bc_space;

/* Globally-shared scalars */
MAIN struct GlobalMemory {
 int open_cells,
     next_res,                /* next reservoir variable for thermalization */
     num_mol,                 /* number of active molecules */
     take_steps,              /* number of steps to take */
     time_step,
     cum_steps,               /* cumulative steps taken */
     num_collisions,          /* molecule collisions in this iteration */
     total_collisions,        /* total molecule collisions */
     BC,                      /* number of boundary collisions */
     res_offset,              /* offset between collided reservoir molecules */
     max_sub;                 /* general purpose max subscript */
 float coll_prob,             /* collision probability */
       free_pop,
       num_flow,
       tot_flt_input;
 long exit_num_flow,
      tot_input;

 /* Barrier and distributed loop and other lock declaraions */
 BARDEC(start_sync)
 BARDEC(end_sync)
 LOCKDEC(output)             /* standard output lock */
 LOCKDEC(collisions_lock)
 LOCKDEC(num_mol_lock)
 LOCKDEC(next_res_lock)
} *Global;

/*
|| Cell Calculation Macros - Could be replaced by more complex
|| routines if grid geometry increases in complexity from 1X1
|| normalized cell.
*/
#define CELL(x) ((int)(x))

/*
 * Random number generators.
 * For mips, MAX_RAND is 32767.
 */
#ifdef ultrix
#  define MAX_RAND 2147483647
   long random();
#else
# ifdef quarks
#   define MAX_RAND 2147483647
    long random();
# else
#   ifdef SVR3
#     define MAX_RAND 2147483647
      long random();
#   else
#     ifdef CMU
#       define MAX_RAND 2147483647
        long random();
#     else
#       define MAX_RAND 32767
#       define random rand
#       define srandom srand
#     endif
#   endif
# endif
#endif

#define RANDOM (random()/(double)MAX_RAND)
#define BOLTZMAN (2.0*(RANDOM+RANDOM+RANDOM+RANDOM+RANDOM+RANDOM)-6.0)

MAIN float space_heightm4,        /* height of active space - 4 */
           space_lengthm4,        /* length of active space - 4 */
           space_widthm4,         /* width of active space - 4 */
           snd_speed,             /* speed of sound */
           fluid_speed,           /* speed of the free stream */
           thermal_factor,
           gama,
           upstream_mfp;          /* mean free path of free stream */
MAIN int I,            /* processor index */
         bc_count,     /* number of boundary conditions defined */
         clump,        /* clumping factor for move loop */
         num_res,      /* number of reservoir molecules */
         num_xcell,    /* X dimension of active space, in cells */
         num_ycell,    /* Y dimension of active space, in cells */
         num_zcell,    /* Z dimension of active space, in cells */
         number_of_processors; 

/* Collision Tables. */
MAIN int Mix_table[MIXS_POSSIBLE][MIX_DOF];
MAIN float Sign_table[SIGNS_POSSIBLE][MIX_DOF];

/* Boundary Conditions Lookup Tables */
MAIN struct bcond {
  float bc_A,
        bc_B,
        bc_C,
        bc_D,
        bc_coef,
        bc_temp;
} *Bconds;

char *calloc(),
     *global_alloc();
void do_command(),
     fill_mix_table(),
     fill_reservoir(), 
     fill_sign_table(), 
     fill_space(), 
     init_variables(), 
     master_advance_solution(), 
     slave_advance_solution();
