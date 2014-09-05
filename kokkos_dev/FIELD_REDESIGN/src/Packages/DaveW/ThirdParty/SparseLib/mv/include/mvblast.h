
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*                                                                           */
/*                                                                           */
/*                   MV++ Numerical Matrix/Vector C++ Library                */
/*                             MV++ Version 1.5                              */
/*                                                                           */
/*                                  R. Pozo                                  */
/*               National Institute of Standards and Technology              */
/*                                                                           */
/*                                  NOTICE                                   */
/*                                                                           */
/* Permission to use, copy, modify, and distribute this software and         */
/* its documentation for any purpose and without fee is hereby granted       */
/* provided that this permission notice appear in all copies and             */
/* supporting documentation.                                                 */
/*                                                                           */
/* Neither the Institution (National Institute of Standards and Technology)  */
/* nor the author makes any representations about the suitability of this    */
/* software for any purpose.  This software is provided ``as is''without     */
/* expressed or implied warranty.                                            */
/*                                                                           */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#ifndef _MV_BLAS1_$TYPE_H_
#define _MV_BLAS1_$TYPE_H_


MV_Vector_$TYPE& operator*=(MV_Vector_$TYPE &x, const $TYPE &a);
MV_Vector_$TYPE operator*(const $TYPE &a, const MV_Vector_$TYPE &x);
MV_Vector_$TYPE operator*(const MV_Vector_$TYPE &x, const $TYPE &a);
MV_Vector_$TYPE operator+(const MV_Vector_$TYPE &x, 
    const MV_Vector_$TYPE &y);
MV_Vector_$TYPE operator-(const MV_Vector_$TYPE &x, 
    const MV_Vector_$TYPE &y);
MV_Vector_$TYPE& operator+=(MV_Vector_$TYPE &x, const MV_Vector_$TYPE &y);
MV_Vector_$TYPE& operator-=(MV_Vector_$TYPE &x, const MV_Vector_$TYPE &y);

$TYPE dot(const MV_Vector_$TYPE &x, const MV_Vector_$TYPE &y);
$TYPE norm(const MV_Vector_$TYPE &x);

#endif

// _MV_BLAS1_$TYPE_H_
