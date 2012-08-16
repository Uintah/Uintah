/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under  
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>

namespace Uintah {

//______________________________________________________________________
//  is_BC_specified--
//    Parses each face in the boundary condition section           
//    of the input file and verifies that each (variable)          
//    has a boundary conditions specified.                         
//______________________________________________________________________
void is_BC_specified(const ProblemSpecP& prob_spec, string variable, const MaterialSubset* matls)
{
  // search the BoundaryConditions problem spec
  // determine if variable bcs have been specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
 
  ProblemSpecP level_ps = grid_ps->findBlock("Level");
  Vector periodic;
  level_ps->getWithDefault("periodic", periodic, Vector(0,0,0));
  if(periodic == Vector(1,1,1)){
    return;
  }
  
  map<string,bool> is_BC_set;     
  map<string,bool> is_periodic;
  is_BC_set["x-"] = false;   is_periodic["x-"] = periodic[0];
  is_BC_set["x+"] = false;   is_periodic["x+"] = periodic[0];
  is_BC_set["y-"] = false;   is_periodic["y-"] = periodic[1];
  is_BC_set["y+"] = false;   is_periodic["y+"] = periodic[1];
  is_BC_set["z-"] = false;   is_periodic["z-"] = periodic[2];
  is_BC_set["z+"] = false;   is_periodic["z+"] = periodic[2];
   
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions"); 
  if(!bc_ps) {
    ostringstream warn;
    warn <<"\n__________________________________\n"               
         << "ERROR: Cannot find the required xml tag"
         << "\n\t <Grid> \n \t\t<BoundaryConditions>\n \t\t</BoundaryConditions>\n\t </Grid>";           
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);  
  }
  
  
  // loop over all faces and determine if a BC has been set
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
   
    map<string,string> face;
    face_ps->getAttributes(face);
   
    // loop over all BCTypes  
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      
      // was the matl specified?
      string id = bc_type["id"];
      int matlIndx = atoi(id.c_str());
      
      bool foundMatl = false;
      if( id == "all" || matls->contains(matlIndx)){
        foundMatl = true;
      }
     
      if ((bc_type["label"] == variable || 
           bc_type["label"] == "Symmetric") &&
           foundMatl ) {
        is_BC_set[face["side"]] = true; 
      }
    }
  }

  //__________________________________
  //Now check if the variable on this face was set
  for( map<string,bool>::iterator iter = is_BC_set.begin();iter !=  is_BC_set.end(); iter++ ){
    string face = (*iter).first;
    bool isSet = (*iter).second;
    
    if (!isSet && !is_periodic[face]){   // BC not set and not periodic
      ostringstream warn;
      warn <<"\n__________________________________\n"  
           << "ERROR: The boundary condition for ( " << variable
           << " ) was not specified on face (" << face 
           << ") for  materialSubset " << *matls << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }
  
  //__________________________________
  // Duplicate periodic BC and normal BCs
  if(periodic.length() != 0){
    bool failed = false;
    string dir = "";
    // periodic and the BC has been set
    if( periodic[0]==1 && ( is_BC_set["x-"] == true || is_BC_set["x+"] == true)){
      dir = "x";
      failed = true;
    }
    if( periodic[1]==1 && ( is_BC_set["y-"] == true || is_BC_set["y+"] == true)){
      dir = dir + ", y";
      failed = true;
    }
    if( periodic[2]==1 && ( is_BC_set["z-"] == true || is_BC_set["x+"] == true)){
      dir = dir + ", z";
      failed = true;
    } 
    
    if( failed ){
      ostringstream warn;
      warn <<"\n__________________________________\n "
           << "ERROR: A periodic AND a normal boundary condition have been specifed for \n"
           << " direction(s): ("<< dir << ")  You can only have one or the other"<< endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);   
    }
  }
}

} // uintah namespace
