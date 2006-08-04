/* $Id$ */


/*
 * Copyright © 2000 The Regents of the University of California. 
 * All Rights Reserved. 
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for educational, research and non-profit purposes, without
 * fee, and without a written agreement is hereby granted, provided that the
 * above copyright notice, this paragraph and the following three paragraphs
 * appear in all copies. 
 *
 * Permission to incorporate this software into commercial products may be
 * obtained by contacting
 * Eric Lund
 * Technology Transfer Office 
 * 9500 Gilman Drive 
 * 411 University Center 
 * University of California 
 * La Jolla, CA 92093-0093
 * (858) 534-0175
 * ericlund@ucsd.edu
 *
 * This software program and documentation are copyrighted by The Regents of
 * the University of California. The software program and documentation are
 * supplied "as is", without any accompanying services from The Regents. The
 * Regents does not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the program was
 * developed for research purposes and is advised not to rely exclusively on
 * the program for any reason. 
 *
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED
 * HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO
 * OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS. 
 */


#ifndef VECTORSEED_H
#define VECTORSEED_H


/*
 * This package defines facilities for dealing with vectors -- arrays that grow
 * and shrink dynamically.  The functionality provided corresponds roughly to
 * that of the Java Vector class.  A vector of ints named foo would be declared
 * like this:
 *
 *    ASV_VECTOR(int) foo = ASV_EMPTY;
 *
 * Any type may be substituted for int, and vectors should always be
 * initialized to ASV_EMPTY.  Initialization is done by a separate macro to
 * allow the use of typedef with vectors, e.g.,
 *
 *    typedef ASV_VECTOR(int) IntVector;
 *    IntVector foo = ASV_EMPTY;
 */


#include <stdlib.h> /* free realloc */
#include <string.h> /* memmove */


/* Declare a vector with elements of type #type#. */
#define ASV_VECTOR(type) \
  struct { \
    unsigned int capacity; \
    unsigned int elementCount; \
    type *elementData; \
    unsigned int increment; \
  }

/* A declaration initializer for a vector that contains no elements. */
#define ASV_EMPTY {0, 0, 0, 1}


/* Adds #element# to the end of #vector#. */
#define ASV_Append(vector,element) \
  do { \
    if((vector).elementCount == (vector).capacity) \
      ASV_AdjustCapacity((vector), (vector).capacity + (vector).increment); \
    (vector).elementData[(vector).elementCount++] = (element); \
  } while(0)


/*
 * Returns the number of elements that may be placed in #vector# before its
 * dynamic space will need to be reallocated.
 */
#define ASV_Capacity(vector) (vector).capacity


/*
 * Returns the #index#'th element of #vector#.  The caller must assure that
 * #index# < ASV_Size(#vector#).
 */
#define ASV_ElementAt(vector,index) (vector).elementData[(index)]


/*
 * Frees the space allocated to #vector#.  To avoid memory leaks, this function
 * must be called after the use of #vector# is complete.
 */
#define ASV_Free(vector) \
  do { \
    if((vector).elementData != 0) \
      free((vector).elementData); \
    (vector).elementData = 0; \
    (vector).elementCount = (vector).capacity = 0; \
  } while(0)


/*
 * Increases the capacity of #vector# to #howBig# if it is not already at least
 * that large.  Capacity increases are normally handled automatically, but this
 * allows making a large increase in preparation for adding many elements.
 */
#define ASV_GrowCapacity(vector,howBig) \
  do { \
    if((vector).capacity < (howBig)) \
      ASV_AdjustCapacity((vector), (howBig)); \
  } while(0)


/*
 * Returns the amount #vector# is expanded when room to add new elements is
 * exhausted.
 */
#define ASV_Increment(vector) (vector).increment


/* Readies #vector# for use; equivalent to assigning ASV_EMPTY. */
#define ASV_Initialize(vector) \
  do { \
    (vector).increment = 1; \
    (vector).elementCount = (vector).capacity = 0; \
    (vector).elementData = 0; \
  } while(0)


/*
 * Inserts #element# as the #index#'th element of #vector#, expanding
 * #vector#'s space if necessary.  A new element may be added in the middle of
 * #vector#, in which case existing elements are shifted to the right.  If
 * #index# > ASV_Size(#vector#) the new element is appended; no "holes" are
 * left in #vector#'s elements.
 */
#define ASV_Insert(vector,element,index) \
  do { \
    if((vector).elementCount == (vector).capacity) \
      ASV_AdjustCapacity((vector), (vector).capacity + (vector).increment); \
    if((index) < (vector).elementCount) { \
      memmove(&(vector).elementData[(index) + 1], \
              &(vector).elementData[(index)], \
              sizeof((vector).elementData[0]) * \
              ((vector).elementCount - (index))); \
      (vector).elementData[(index)] = (element); \
    } \
    else \
      (vector).elementData[(vector).elementCount] = (element); \
    (vector).elementCount++; \
  } while(0)
   

/*
 * Removes the #index#'th element of #vector#.  The caller must assure that
 * #index# < ASV_Size(#vector#).
 */
#define ASV_Remove(vector,index) \
  do { \
    if((index) < --(vector).elementCount) \
      memmove(&(vector).elementData[(index)], \
              &(vector).elementData[(index) + 1], \
              sizeof((vector).elementData[0]) * \
              ((vector).elementCount - (index))); \
  } while(0)


/* Sets the capacity of #vector# to exactly #howBig# elements. */
#define ASV_SetCapacity(vector,howBig) \
  do { \
    if((vector).capacity != (howBig)) { \
       ASV_AdjustCapacity((vector), (howBig)); \
       ASV_ShrinkSize((vector), (howBig)); \
    } \
  } while(0)


/*
 * Sets the amount #vector# is expanded when room to add new elements is
 * exhausted to #howMany#.  The default is 1.
 */
#define ASV_SetIncrement(vector,howMany) (vector).increment = (howMany)


/*
 * Reduces the capacity of #vector# to #howBig# if it is not already at least
 * that small.
 */
#define ASV_ShrinkCapacity(vector,howBig) \
  do { \
    if((vector).capacity > (howBig)) { \
      ASV_AdjustCapacity((vector), (howBig)); \
      ASV_ShrinkSize((vector), (howBig)); \
    } \
  } while(0)


/* Discards any elements of #vector# beyond the #howBig#'th one. */
#define ASV_ShrinkSize(vector,howBig) \
  do { \
    if((vector).elementCount > (howBig)) \
      (vector).elementCount = (howBig); \
  } while(0)


/* Returns the number of elements in #vector#. */
#define ASV_Size(vector) (vector).elementCount


/*
 * Set the capacity of #vector# to #howBig#.  This macro is solely for the use
 * of the other package macros and should never be called directly.
 */
#define ASV_AdjustCapacity(vector,howBig) \
  do { \
    (vector).capacity = (howBig); \
    *((void **)&(vector).elementData) = realloc \
      ((vector).elementData, (vector).capacity * sizeof((vector).elementData[0])); \
  } while(0)


#ifdef ASV_SHORT_NAMES

#define VECTOR ASV_VECTOR
#define EMPTY ASV_EMPTY

#define Append ASV_Append
#define Capacity ASV_Capacity
#define ElementAt ASV_ElementAt
#define Free ASV_Free
#define GrowCapacity ASV_GrowCapacity
#define Increment ASV_Increment
#define Initialize ASV_Initialize
#define Insert ASV_Insert
#define Remove ASV_Remove
#define SetCapacity ASV_SetCapacity
#define SetIncrement ASV_SetIncrement
#define ShrinkCapacity ASV_ShrinkCapacity
#define ShrinkSize ASV_ShrinkSize
#define Size ASV_Size
#define AdjustCapacity ASV_AdjustCapacity

#endif


#endif
