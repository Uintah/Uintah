#ifndef NRRD_MACROS_HAS_BEEN_INCLUDED
#define NRRD_MACROS_HAS_BEEN_INCLUDED

/*
******** NRRD_MAX(a,b), NRRD_MIN(a,b), NRRD_ABS(a)
**
** the usual
*/
#define NRRD_MAX(a,b) ((a) > (b) ? (a) : (b))
#define NRRD_MIN(a,b) ((a) > (b) ? (b) : (a))
#define NRRD_ABS(a) ((a) > 0 ? (a) : -(a))

/*
******** NRRD_ROUND(x)
**
** rounds argument to nearest integer
** (using NRRD_BIGINT to make sure value isn't truncated unnecessarily)
*/
#define NRRD_ROUND(x) (NRRD_BIG_INT)((x) + 0.5)

/*
******** NRRD_LERP(f,x,y)
**
** linear interpolation between x and y, weighted by f.
** Returns x when f = 0;
*/
#define NRRD_LERP(f,x,y) ((x) + (f)*((y) - (x)))

/*
******** NRRD_EXISTS(x)
**
** x is not nan
** If simple arithmetic trick (x - x != 0) doesn't work, need to
** define this as a wrapper around nrrdIsNand()
*/
/* #define NRRD_EXISTS(x) (!((x) - (x))) */
#define NRRD_EXISTS(x) (!nrrdIsNand((x)))

/*
******** NRRD_INSIDE(a,b,c)
**
** is true if the middle argument is in the closed interval
** defined by the first and third arguments
*/
#define NRRD_INSIDE(a,b,c) ((a) <= (b) && (b) <= (c))

/*
******** NRRD_CLAMP(a,b,c)
**
** returns the middle argument, after being clamped to the closed
** interval defined by the first and third arguments
*/
#define NRRD_CLAMP(a,b,c) ((b) < (a)        \
		 	   ? (a)            \
			   : ((b) > (c)     \
			      ? (c)         \
			      : (b)))

/*
******** NRRD_AFFINE(i,x,I,o,O)
**
** given intervals [i,I], [o,O] and a value x which may or may not be
** inside [i,I], return the value y such that y stands in the same
** relationship to [o,O] that x does with [i,I].  Or:
**
**    y - o         x - i     
**   -------   =   -------
**    O - o         I - i
**
** It is the callers responsibility to make sure I-i and O-o are 
** both greater than zero
*/
#define NRRD_AFFINE(i,x,I,o,O) ( ((double)(O)-(o))*((double)(x)-(i)) \
		 	        / ((double)(I)-(i)) + (o))

/*
******** NRRD_INDEX(i,x,I,L,t)
**
** READ CAREFULLY!!
**
** Utility for mapping a floating point x in given range [i,I] to the
** index of an array with L elements, AND SAVES THE INDEX INTO GIVEN
** VARIABLE t, WHICH MUST BE OF TYPE INT (or long int, etc) because
** this relies on the implicit cast.  ALSO, t must be of a type large
** enough to hold ONE GREATER than L.  So you can't pass a variable
** of type unsigned char if L is 256
**
** DOES NOT DO BOUNDS CHECKING: given an x which is not inside [i,I],
** this will produce an index not inside [0,L].  The mapping is
** accomplished by dividing the range from i to I into L intervals,
** all but the last of which is half-open; the last one is closed.
** For example, the number line from 0 to 3 would be divided as
** follows for a call with i = 0, I = 4, L = 4:
**
** index:       0    1    2    3 = L-1
** intervals: [   )[   )[   )[    ]
**            |----|----|----|----|
** value:     0    1    2    3    4
**
** The main point of the diagram above is to show how I made the
** arbitrary decision to orient the half-open interval, and which
** end has the closed interval.
**
** Note that NRRD_INDEX(0,3,4,4,t) and NRRD_INDEX(0,4,4,4,t) both set t = 3
**
** The reason that this macro requires a argument for saving the
** result is that this is the easiest way to avoid extra conditionals.
** Otherwise, we'd have to do some check to see if x is close enough
** to I so that the generated index would be L and not L-1.  "Close
** enough" because due to precision problems you can have an x < I
** such that (x-i)/(I-i) == 1.  Simplest to just do the index generation
** and then do a hack to see if the index is too large by 1.
*/
#define NRRD_INDEX(i,x,I,L,t) ((t) = (L)*((double)(x)-(i))/((double)(I)-(i)), \
			       (t) -= ((t) == (L)))

/*
******** NRRD_CLAMP_INDEX(i,x,I,L,t)
**
** variant of NRRD_INDEX which does bounds checking, and is thereby slower
*/
#define NRRD_CLAMP_INDEX(i,x,I,L,t) (t=((L)*((double)(NRRDCLAMP(i,x,I))-(i)) \
					 /((double)(I)-(i))), \
				     t -= (t == L))

#endif /* NRRD_MACROS_HAS_BEEN_INCLUDED */
