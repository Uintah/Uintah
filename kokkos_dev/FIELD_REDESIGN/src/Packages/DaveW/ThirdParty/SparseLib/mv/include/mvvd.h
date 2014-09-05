
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
//      mv_vector_double.h       Basic vector class (double precision)
//

#ifndef _MV_VECTOR_double_H
#define _MV_VECTOR_double_H    



#include <stdlib.h>
#include <iostream>       // for formatted printing of matrices

using namespace std;


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

class MV_Vector_double
{                                                                      
    protected:                                                           
           double *p_;
           unsigned int dim_;
           int ref_;  // 0 or 1; does this own its own memory space?
    public:                                                            


        /*::::::::::::::::::::::::::*/                                 
        /* Constructors/Destructors */                                 
        /*::::::::::::::::::::::::::*/                                 

  
    MV_Vector_double();                             
    MV_Vector_double(unsigned int);                             
    MV_Vector_double(unsigned int, const double&);   
                                                     
                                                    
    MV_Vector_double(double*, unsigned int);      
    MV_Vector_double(const double*, unsigned int);        
    MV_Vector_double(const MV_Vector_double &); 
    
    // reference of an exisiting data structure
    //
    // note that ref() is initalized with i rather than 1.
    // this is so compilers will not generate a warning that i was
    // not used in the construction.  (MV_Vector::ref_type is an enum that
    // can *only* have the value of 1.
    //
    MV_Vector_double(double* d, unsigned int N, MV_Vector_::ref_type i) :
                            p_(d), dim_(N), ref_(i) {}

    MV_Vector_double(const MV_Vector_double &V, MV_Vector_::ref_type i)   :
                            p_(V.p_), dim_(V.dim_), ref_(i) {}

    ~MV_Vector_double();                              
                                                                       
        /*::::::::::::::::::::::::::::::::*/                           
        /*  Indices and access operations */                           
        /*::::::::::::::::::::::::::::::::*/                           
                                                                       

    double&      operator()(unsigned int i)
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }
    const  double&       operator()(unsigned int i) const 
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }

    double&      operator[](unsigned int i)
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }
    const  double&       operator[](unsigned int i) const 
                  {
#                   ifdef MV_VECTOR_BOUNDS_CHECK
                    assert(i < dim_);
#                   endif
                    return p_[i];
                  }


    MV_Vector_double operator()(const MV_VecIndex &I) ;
    MV_Vector_double operator()(void);
    const MV_Vector_double operator()(void) const;
    const MV_Vector_double operator()(const MV_VecIndex &I) const;

    inline unsigned int             size() const { return dim_;}
    inline unsigned int             dim() const { return dim_;}
    inline int                      ref() const { return  ref_;}
    inline int                      null() const {return dim_== 0;}
            //
            // Create a new *uninitalized* vector of size N
            MV_Vector_double & newsize(unsigned int );
                                                                       
        /*::::::::::::::*/                                             
        /*  Assignment  */                                             
        /*::::::::::::::*/                                             
                                                                       
    MV_Vector_double & operator=(const MV_Vector_double&);
    MV_Vector_double & operator=(const double&);


    friend ostream& operator<<(ostream &s, const MV_Vector_double &A);

};                                                                     



#endif  
