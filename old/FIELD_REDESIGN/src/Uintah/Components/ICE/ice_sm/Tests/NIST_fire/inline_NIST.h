#ifndef __INLINE_H
#define __INLINE_H
#ifdef __sgi
    #define inline __inline
#endif
/* ---------------------------------------------------------------------
 Function:  burnoutfunction
 Filename:  inline_NIST.h
 Purpose:   This is a user defined function that computes the burnout
            history of the thermal elements
            
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       5/22/00       written 
             
 Implementation Note:
            This function has been inlined to speed it up since it is called 
            for TE.                            
 ---------------------------------------------------------------------  */        
inline double burnout_function(        
        double  t,
        double  t_burnout_TE)             
{ 
    double temp;         
      temp =    1.0 - t/t_burnout_TE;
    return temp;
}
#endif
