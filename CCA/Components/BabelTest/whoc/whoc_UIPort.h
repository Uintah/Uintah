/*
 * File:          whoc_UIPort.h
 * Symbol:        whoc.UIPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20030305 18:50:11 MST
 * Generated:     20030305 18:50:17 MST
 * Description:   Client-side glue code for whoc.UIPort
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 10
 * source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/whoc/whoc.sidl
 */

#ifndef included_whoc_UIPort_h
#define included_whoc_UIPort_h

/**
 * Symbol "whoc.UIPort" (version 1.0)
 */
struct whoc_UIPort__object;
struct whoc_UIPort__array;
typedef struct whoc_UIPort__object* whoc_UIPort;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
whoc_UIPort
whoc_UIPort__create(void);

/**
 * &amp;lt;p&amp;gt;
 * Add one to the intrinsic reference count in the underlying object.
 * Object in &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * &amp;lt;/p&amp;gt;
 * &amp;lt;p&amp;gt;
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * &amp;lt;/p&amp;gt;
 */
void
whoc_UIPort_addReference(
  whoc_UIPort self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
whoc_UIPort_deleteReference(
  whoc_UIPort self);

/**
 * Return true if and only if &amp;lt;code&amp;gt;obj&amp;lt;/code&amp;gt; refers to the same
 * object as this object.
 */
SIDL_bool
whoc_UIPort_isSame(
  whoc_UIPort self,
  SIDL_BaseInterface iobj);

/**
 * Check whether the object can support the specified interface or
 * class.  If the &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; type name in &amp;lt;code&amp;gt;name&amp;lt;/code&amp;gt;
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling &amp;lt;code&amp;gt;deleteReference&amp;lt;/code&amp;gt; on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
SIDL_BaseInterface
whoc_UIPort_queryInterface(
  whoc_UIPort self,
  const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; type name.  This
 * routine will return &amp;lt;code&amp;gt;true&amp;lt;/code&amp;gt; if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
whoc_UIPort_isInstanceOf(
  whoc_UIPort self,
  const char* name);

/**
 * Execute some encapsulated functionality on the component. 
 * @return 0 if ok, -1 if internal error but component may be used further,
 * -2 if error so severe that component cannot be further used safely.
 */
int32_t
whoc_UIPort_ui(
  whoc_UIPort self);

/**
 * Cast method for interface and class type conversions.
 */
whoc_UIPort
whoc_UIPort__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
whoc_UIPort__cast2(
  void* obj,
  const char* type);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_createCol(int32_t        dimen,
                             const int32_t lower[],
                             const int32_t upper[]);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_createRow(int32_t        dimen,
                             const int32_t lower[],
                             const int32_t upper[]);

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_create1d(int32_t len);

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteReference will be called on the
 * value being replaced if it is not NULL.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_borrow(whoc_UIPort*firstElement,
                          int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_smartCopy(struct whoc_UIPort__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
whoc_UIPort__array_addReference(struct whoc_UIPort__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
whoc_UIPort__array_deleteReference(struct whoc_UIPort__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
whoc_UIPort
whoc_UIPort__array_get1(const struct whoc_UIPort__array* array,
                        const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
whoc_UIPort
whoc_UIPort__array_get2(const struct whoc_UIPort__array* array,
                        const int32_t i1,
                        const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
whoc_UIPort
whoc_UIPort__array_get3(const struct whoc_UIPort__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
whoc_UIPort
whoc_UIPort__array_get4(const struct whoc_UIPort__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        const int32_t i4);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
whoc_UIPort
whoc_UIPort__array_get(const struct whoc_UIPort__array* array,
                       const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
whoc_UIPort__array_set1(struct whoc_UIPort__array* array,
                        const int32_t i1,
                        whoc_UIPort const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
whoc_UIPort__array_set2(struct whoc_UIPort__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        whoc_UIPort const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
whoc_UIPort__array_set3(struct whoc_UIPort__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        whoc_UIPort const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
whoc_UIPort__array_set4(struct whoc_UIPort__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        const int32_t i4,
                        whoc_UIPort const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
whoc_UIPort__array_set(struct whoc_UIPort__array* array,
                       const int32_t indices[],
                       whoc_UIPort const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
whoc_UIPort__array_dimen(const struct whoc_UIPort__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
whoc_UIPort__array_lower(const struct whoc_UIPort__array* array,
                         const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
whoc_UIPort__array_upper(const struct whoc_UIPort__array* array,
                         const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
whoc_UIPort__array_stride(const struct whoc_UIPort__array* array,
                          const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
whoc_UIPort__array_isColumnOrder(const struct whoc_UIPort__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
whoc_UIPort__array_isRowOrder(const struct whoc_UIPort__array* array);

/**
 * Copy the contents of one array (src) to a second array
 * (dest). For the copy to take place, both arrays must
 * exist and be of the same dimension. This method will
 * not modify dest's size, index bounds, or stride; only
 * the array element values of dest may be changed by
 * this function. No part of src is ever changed by copy.
 * 
 * On exit, dest[i][j][k]... = src[i][j][k]... for all
 * indices i,j,k...  that are in both arrays. If dest and
 * src have no indices in common, nothing is copied. For
 * example, if src is a 1-d array with elements 0-5 and
 * dest is a 1-d array with elements 2-3, this function
 * will make the following assignments:
 *   dest[2] = src[2],
 *   dest[3] = src[3].
 * The function copied the elements that both arrays have
 * in common.  If dest had elements 4-10, this function
 * will make the following assignments:
 *   dest[4] = src[4],
 *   dest[5] = src[5].
 */
void
whoc_UIPort__array_copy(const struct whoc_UIPort__array* src,
                              struct whoc_UIPort__array* dest);

/**
 * If necessary, convert a general matrix into a matrix
 * with the required properties. This checks the
 * dimension and ordering of the matrix.  If both these
 * match, it simply returns a new reference to the
 * existing matrix. If the dimension of the incoming
 * array doesn't match, it returns NULL. If the ordering
 * of the incoming array doesn't match the specification,
 * a new array is created with the desired ordering and
 * the content of the incoming array is copied to the new
 * array.
 * 
 * The ordering parameter should be one of the constants
 * defined in enum SIDL_array_ordering
 * (e.g. SIDL_general_order, SIDL_column_major_order, or
 * SIDL_row_major_order). If you specify
 * SIDL_general_order, this routine will only check the
 * dimension because any matrix is SIDL_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct whoc_UIPort__array*
whoc_UIPort__array_ensure(struct whoc_UIPort__array* src,
                          int32_t dimen,
int     ordering);

#ifdef __cplusplus
}
#endif
#endif
