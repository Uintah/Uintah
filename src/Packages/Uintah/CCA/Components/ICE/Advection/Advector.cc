#include <Packages/Uintah/CCA/Components/ICE/Advection/Advector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>

using namespace Uintah;

Advector::Advector()
{
    //__________________________________
    //   S L A B S

    OF_slab[RIGHT] = RIGHT;         IF_slab[RIGHT]  = LEFT;
    OF_slab[LEFT]  = LEFT;          IF_slab[LEFT]   = RIGHT;
    OF_slab[TOP]   = TOP;           IF_slab[TOP]    = BOTTOM;
    OF_slab[BOTTOM]= BOTTOM;        IF_slab[BOTTOM] = TOP;  
    OF_slab[FRONT] = FRONT;         IF_slab[FRONT]  = BACK;
    OF_slab[BACK]  = BACK;          IF_slab[BACK]   = FRONT;   

    // Slab adjacent cell
    S_ac[RIGHT]  =  IntVector( 1, 0, 0);   
    S_ac[LEFT]   =  IntVector(-1, 0, 0);   
    S_ac[TOP]    =  IntVector( 0, 1, 0);   
    S_ac[BOTTOM] =  IntVector( 0,-1, 0);   
    S_ac[FRONT]  =  IntVector( 0, 0, 1);   
    S_ac[BACK]   =  IntVector( 0, 0,-1);   
    
    //__________________________________
    //   E D G E S   OF = Outflux IF = influx 
    //RIGHT FACE
    OF_edge[RIGHT][0] = TOP_R;      IF_edge[RIGHT][0] = BOT_L;   
    OF_edge[RIGHT][1] = BOT_R;      IF_edge[RIGHT][1] = TOP_L;   
    OF_edge[RIGHT][2] = RIGHT_FR;   IF_edge[RIGHT][2] = LEFT_BK;
    OF_edge[RIGHT][3] = RIGHT_BK;   IF_edge[RIGHT][3] = LEFT_FR; 

    // LEFT FACE
    OF_edge[LEFT][0] = TOP_L;       IF_edge[LEFT][0] = BOT_R;   
    OF_edge[LEFT][1] = BOT_L;       IF_edge[LEFT][1] = TOP_R;   
    OF_edge[LEFT][2] = LEFT_FR;     IF_edge[LEFT][2] = RIGHT_BK;
    OF_edge[LEFT][3] = LEFT_BK;     IF_edge[LEFT][3] = RIGHT_FR;

    // TOP FACE
    OF_edge[TOP][0] = TOP_R;        IF_edge[TOP][0] = BOT_L;  
    OF_edge[TOP][1] = TOP_L;        IF_edge[TOP][1] = BOT_R; 
    OF_edge[TOP][2] = TOP_FR;       IF_edge[TOP][2] = BOT_BK;
    OF_edge[TOP][3] = TOP_BK;       IF_edge[TOP][3] = BOT_FR;

    // BOTTOM FACE
    OF_edge[BOTTOM][0] = BOT_R;     IF_edge[BOTTOM][0] = TOP_L;
    OF_edge[BOTTOM][1] = BOT_L;     IF_edge[BOTTOM][1] = TOP_R;   
    OF_edge[BOTTOM][2] = BOT_FR;    IF_edge[BOTTOM][2] = TOP_BK;                        
    OF_edge[BOTTOM][3] = BOT_BK;    IF_edge[BOTTOM][3] = TOP_FR;                        

    // FRONT FACE
    OF_edge[FRONT][0] = RIGHT_FR;   IF_edge[FRONT][0] = LEFT_BK;  
    OF_edge[FRONT][1] = LEFT_FR;    IF_edge[FRONT][1] = RIGHT_BK; 
    OF_edge[FRONT][2] = TOP_FR;     IF_edge[FRONT][2] = BOT_BK;                          
    OF_edge[FRONT][3] = BOT_FR;     IF_edge[FRONT][3] = TOP_BK;                          

    // BACK FACE
    OF_edge[BACK][0] = RIGHT_BK;    IF_edge[BACK][0] = LEFT_FR;  
    OF_edge[BACK][1] = LEFT_BK;     IF_edge[BACK][1] = RIGHT_FR; 
    OF_edge[BACK][2] = TOP_BK;      IF_edge[BACK][2] = BOT_FR;
    OF_edge[BACK][3] = BOT_BK;      IF_edge[BACK][3] = TOP_FR; 

    // Adjacent edge cells.
    E_ac[RIGHT][0]  =  IntVector( 1, 1, 0);   //TOP_R     BOT_L 
    E_ac[RIGHT][1]  =  IntVector( 1,-1, 0);   //BOT_R     TOP_L
    E_ac[RIGHT][2]  =  IntVector( 1, 0, 1);   //RIGHT_FR  LEFT_BK
    E_ac[RIGHT][3]  =  IntVector( 1, 0,-1);   //RIGHT_BK  LEFT_FR

    E_ac[LEFT][0]   =  IntVector(-1, 1, 0);   //TOP_L;    BOT_R;    
    E_ac[LEFT][1]   =  IntVector(-1,-1, 0);   //BOT_L;    TOP_R;    
    E_ac[LEFT][2]   =  IntVector(-1, 0, 1);   //LEFT_FR;  RIGHT_BK; 
    E_ac[LEFT][3]   =  IntVector(-1, 0,-1);   //LEFT_BK;  RIGHT_FR; 

    E_ac[TOP][0]    =  IntVector( 1, 1, 0);   //TOP_R;   BOT_L; 
    E_ac[TOP][1]    =  IntVector(-1, 1, 0);   //TOP_L;   BOT_R; 
    E_ac[TOP][2]    =  IntVector( 0, 1, 1);   //TOP_FR;  BOT_BK;
    E_ac[TOP][3]    =  IntVector( 0, 1,-1);   //TOP_BK;  BOT_FR; 

    E_ac[BOTTOM][0] =  IntVector( 1,-1, 0);   //BOT_R;   TOP_L;    
    E_ac[BOTTOM][1] =  IntVector(-1,-1, 0);   //BOT_L;   TOP_R;    
    E_ac[BOTTOM][2] =  IntVector( 0,-1, 1);   //BOT_FR;  TOP_BK;   
    E_ac[BOTTOM][3] =  IntVector( 0,-1,-1);   //BOT_BK;  TOP_FR;   

    E_ac[FRONT][0]  =  IntVector( 1, 0, 1);   //RIGHT_FR  LEFT_BK;    
    E_ac[FRONT][1]  =  IntVector(-1, 0, 1);   //LEFT_FR;  RIGHT_BK;   
    E_ac[FRONT][2]  =  IntVector( 0, 1, 1);   //TOP_FR    BOT_BK;     
    E_ac[FRONT][3]  =  IntVector( 0,-1, 1);   //BOT_FR;   TOP_BK;   

    E_ac[BACK][0]   =  IntVector( 1, 0,-1);   //RIGHT_BK;  LEFT_FR;  
    E_ac[BACK][1]   =  IntVector(-1, 0,-1);   //LEFT_BK;   RIGHT_FR  
    E_ac[BACK][2]   =  IntVector( 0, 1,-1);   //TOP_BK     BOT_FR;   
    E_ac[BACK][3]   =  IntVector( 0,-1,-1);   //BOT_BK;    TOP_FR;     
    
         
    //__________________________________
    //  C O R N E R S
       
    // RIGHT FACE
    OF_corner[RIGHT][0] = TOP_R_BK;   IF_corner[RIGHT][0] = BOT_L_FR;       
    OF_corner[RIGHT][1] = TOP_R_FR;   IF_corner[RIGHT][1] = BOT_L_BK;       
    OF_corner[RIGHT][2] = BOT_R_BK;   IF_corner[RIGHT][2] = TOP_L_FR;   
    OF_corner[RIGHT][3] = BOT_R_FR;   IF_corner[RIGHT][3] = TOP_L_BK;   

    // LEFT FACE
    OF_corner[LEFT][0] = TOP_L_BK;    IF_corner[LEFT][0] = BOT_R_FR;      
    OF_corner[LEFT][1] = TOP_L_FR;    IF_corner[LEFT][1] = BOT_R_BK;      
    OF_corner[LEFT][2] = BOT_L_BK;    IF_corner[LEFT][2] = TOP_R_FR;  
    OF_corner[LEFT][3] = BOT_L_FR;    IF_corner[LEFT][3] = TOP_R_BK;  

    // TOP FACE
    OF_corner[TOP][0] = TOP_R_BK;     IF_corner[TOP][0] = BOT_L_FR;     
    OF_corner[TOP][1] = TOP_R_FR;     IF_corner[TOP][1] = BOT_L_BK;    
    OF_corner[TOP][2] = TOP_L_BK;     IF_corner[TOP][2] = BOT_R_FR;    
    OF_corner[TOP][3] = TOP_L_FR;     IF_corner[TOP][3] = BOT_R_BK;    

    // BOTTOM FACE
    OF_corner[BOTTOM][0] = BOT_R_BK;  IF_corner[BOTTOM][0] = TOP_L_FR;    
    OF_corner[BOTTOM][1] = BOT_R_FR;  IF_corner[BOTTOM][1] = TOP_L_BK;      
    OF_corner[BOTTOM][2] = BOT_L_BK;  IF_corner[BOTTOM][2] = TOP_R_FR;                           
    OF_corner[BOTTOM][3] = BOT_L_FR;  IF_corner[BOTTOM][3] = TOP_R_BK;                           

    // FRONT FACE
    OF_corner[FRONT][0] = TOP_R_FR;   IF_corner[FRONT][0] = BOT_L_BK;  
    OF_corner[FRONT][1] = BOT_R_FR;   IF_corner[FRONT][1] = TOP_L_BK; 
    OF_corner[FRONT][2] = TOP_L_FR;   IF_corner[FRONT][2] = BOT_R_BK;                          
    OF_corner[FRONT][3] = BOT_L_FR;   IF_corner[FRONT][3] = TOP_R_BK;                          

    // BACK FACE
    OF_corner[BACK][0] = TOP_R_BK;    IF_corner[BACK][0] = BOT_L_FR; 
    OF_corner[BACK][1] = BOT_R_BK;    IF_corner[BACK][1] = TOP_L_FR; 
    OF_corner[BACK][2] = TOP_L_BK;    IF_corner[BACK][2] = BOT_R_FR;
    OF_corner[BACK][3] = BOT_L_BK;    IF_corner[BACK][3] = TOP_R_FR;    
    
    // Adjacent edge cells.

    C_ac[RIGHT][0]  = IntVector( 1, 1,-1);   //TOP_R_BK  BOT_L_FR
    C_ac[RIGHT][1]  = IntVector( 1, 1, 1);   //TOP_R_FR  BOT_L_BK
    C_ac[RIGHT][2]  = IntVector( 1,-1,-1);   //BOT_R_BK  TOP_L_FR
    C_ac[RIGHT][3]  = IntVector( 1,-1, 1);   //BOT_R_FR  TOP_L_BK
    
    C_ac[LEFT][0]   = IntVector(-1, 1,-1);   //TOP_L_BK  BOT_R_FR 
    C_ac[LEFT][1]   = IntVector(-1, 1, 1);   //TOP_L_FR  BOT_R_BK 
    C_ac[LEFT][2]   = IntVector(-1,-1,-1);   //BOT_L_BK  TOP_R_FR 
    C_ac[LEFT][3]   = IntVector(-1,-1, 1);   //BOT_L_FR  TOP_R_BK 

    C_ac[TOP][0]    = IntVector( 1, 1,-1);   //TOP_R_BK  BOT_L_FR
    C_ac[TOP][1]    = IntVector( 1, 1, 1);   //TOP_R_FR  BOT_L_BK
    C_ac[TOP][2]    = IntVector(-1, 1,-1);   //TOP_L_BK  BOT_R_FR
    C_ac[TOP][3]    = IntVector(-1, 1, 1);   //TOP_L_FR  BOT_R_BK

    C_ac[BOTTOM][0] = IntVector( 1,-1,-1);   //BOT_R_BK  TOP_L_FR   
    C_ac[BOTTOM][1] = IntVector( 1,-1, 1);   //BOT_R_FR  TOP_L_BK   
    C_ac[BOTTOM][2] = IntVector(-1,-1,-1);   //BOT_L_BK  TOP_R_FR   
    C_ac[BOTTOM][3] = IntVector(-1,-1, 1);   //BOT_L_FR  TOP_R_BK   

    C_ac[FRONT][0]  = IntVector( 1, 1, 1);   //TOP_R_FR  BOT_L_BK   
    C_ac[FRONT][1]  = IntVector( 1,-1, 1);   //BOT_R_FR  TOP_L_BK   
    C_ac[FRONT][2]  = IntVector(-1, 1, 1);   //TOP_L_FR  BOT_R_BK   
    C_ac[FRONT][3]  = IntVector(-1,-1, 1);   //BOT_L_FR  TOP_R_BK 

    C_ac[BACK][0]   = IntVector( 1, 1,-1);   //TOP_R_BK  BOT_L_FR  
    C_ac[BACK][1]   = IntVector( 1,-1,-1);   //BOT_R_BK  TOP_L_FR  
    C_ac[BACK][2]   = IntVector(-1, 1,-1);   //TOP_L_BK  BOT_R_FR  
    C_ac[BACK][3]   = IntVector(-1,-1,-1);   //BOT_L_BK  TOP_R_FR
}

Advector::~Advector()
{
}



//______________________________________________________________________
//
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif  

//______________________________________________________________________
//  
namespace Uintah {

  static MPI_Datatype makeMPI_fflux()
  {
    ASSERTEQ(sizeof(fflux), sizeof(double)*6);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }

  const TypeDescription* fun_getTypeDescription(fflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "fflux", true, 
				  &makeMPI_fflux);
    }
    return td;
  }
  
  static MPI_Datatype makeMPI_eflux()
  {
    ASSERTEQ(sizeof(eflux), sizeof(double)*12);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 12, 12, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(eflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "eflux", true, 
                              &makeMPI_eflux);
    }
    return td;
  }
  
  static MPI_Datatype makeMPI_cflux()
  {
    ASSERTEQ(sizeof(cflux), sizeof(double)*8);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 8, 8, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(cflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                              "cflux", true, 
                              &makeMPI_cflux);
    }
    return td;
  }
  
} // namespace Uintah


//______________________________________________________________________
//  
namespace SCIRun {

  void swapbytes( Uintah::fflux& f) {
    double *p = f.d_fflux;
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  }

  void swapbytes( Uintah::eflux& e) {
    double *p = e.d_eflux;
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  }
  
  void swapbytes( Uintah::cflux& c) {
    double *p = c.d_cflux;
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  }

} // namespace SCIRun
