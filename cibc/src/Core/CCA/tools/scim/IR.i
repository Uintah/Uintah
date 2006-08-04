
%module scim 

%typemap(in) char *[] {
  //Get the length of the array 
  int size = RARRAY($input)->len;     
  int i;
  $1 = (char **) malloc((size+1)*sizeof(char *));
  //Get the first element in memory 
  VALUE *ptr = RARRAY($input)->ptr;   
  for (i=0; i < size; i++, ptr++)
    //Convert Ruby Object String to char*
    $1[i]= STR2CSTR(*ptr); 
  $1[i]=NULL;  //End of list 
}

%typemap(freearg) char *[] {
  free((char *) $1);
}

%{
#include "IR.h"
%}
                                                                                                                                                 
%include "std_string.i"
%include "std_vector.i"
%include "cpointer.i"

//Vector instantiation definitions
namespace std {
    %template(MapVector)    vector<IrMap* >;
}


//Pointer manipulation types
%pointer_class(IrMap, IrMap_p);
%pointer_class(IrMethodMap, IrMethodMap_p);


%include "IR.h"


