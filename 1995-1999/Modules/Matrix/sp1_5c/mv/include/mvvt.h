
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

//
//      mv_vector_$TYPE.h       Basic vector class ($TYPE precision)
//

#ifndef _MV_VECTOR_$TYPE_H
#define _MV_VECTOR_$TYPE_H    



#include <stdlib.h>
#include <iostream.h>       // for formatted printing of matrices


#ifdef MV_VECTOR_BOUNDS_CHECK
#   include <assert.h>
#endif

#include "mvvind.h"

// this is really used as a sort of global constant. The reason
// for creating its own type is that so it can be overloaded to perform
// a deep or shallow assignement.  (Any variable of type MV_Vector_::ref_type
// has only one possible value: one.)
//   It is included as a seperate file to avoid multiple definitions.

#include "mvvrf.h"

class MV_Vector_$TYPE
{                                                                      
    protected:                                                           
           $TYPE *p_;
           unsigned int dim_;
           int ref_;  // 0 or 1; does this own its own memory space?
    public:                                                            


        /*::::::::::::::::::::::::::*/                                 
        /* Constructors/Destructors */                                 
        /*::::::::::::::::::::::::::*/                                 
                                                                       
    MV_Vector_$TYPE();                             
    MV_Vector_$TYPE(unsigned int);                             
    MV_Vector_$TYPE(unsigned int, const $TYPE&);   
                                                     
                                                    
    MV_Vector_$TYPE($TYPE*, unsigned int);      
    MV_Vector_$TYPE(const $TYPE*, unsigned int);        
    MV_Vector_$TYPE(const MV_Vector_$TYPE &); 
    
    // reference of an exisiting data structure
    //
    // note that ref() is initalized with i rather than 1.
    // this is so compilers will not generate a warning that i was
    // not used in the construction.  (MV_Vector::ref_type is an enum that
    // can *only* have the value of 1.
    //
    MV_Vector_$TYPE($TYPE* d, unsigned int N, MV_Vector_::ref_type i) :
                            p_(d), dim_(N), ref_(i) {}

    MV_Vector_$TYPE(const MV_Vector_$TYPE &V, MV_Vector_::ref_type i)   :
                            p_(V.p_), dim_(V.dim_), ref_(i) {}

    ~MV_Vector_$TYPE();                              
                                                                       
        /*::::::::::::::::::::::::::::::::*/                           
        /*  Indices and access operations */                           
        /*::::::::::::::::::::::::::::::::*/                           
                                                                       

    $TYPE&      operator()(unsigned int i)
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }
    const  $TYPE&       operator()(unsigned int i) const 
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }

    $TYPE&      operator[](unsigned int i)
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }
    const  $TYPE&       operator[](unsigned int i) const 
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }


    MV_Vector_$TYPE operator()(const MV_VecIndex &I) ;
    MV_Vector_$TYPE operator()(void);
    const MV_Vector_$TYPE operator()(void) const;
    const MV_Vector_$TYPE operator()(const MV_VecIndex &I) const;

    inline unsigned int             size() const { return dim_;}
    inline unsigned int             dim() const { return dim_;}
    inline int                      ref() const { return  ref_;}
    inline int                      null() const {return dim_== 0;}
            //
            // Create a new *uninitalized* vector of size N
            MV_Vector_$TYPE & newsize(unsigned int );
                                                                       
        /*::::::::::::::*/                                             
        /*  Assignment  */                                             
        /*::::::::::::::*/                                             
                                                                       
    MV_Vector_$TYPE & operator=(const MV_Vector_$TYPE&);
    MV_Vector_$TYPE & operator=(const $TYPE&);


    friend ostream& operator<<(ostream &s, const MV_Vector_$TYPE &A);

};                                                                     



#endif  
