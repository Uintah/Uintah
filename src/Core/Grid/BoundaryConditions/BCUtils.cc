/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/BoundaryConditions/BoundCondReader.h>

using namespace std;

namespace Uintah {

  //______________________________________________________________________
  //  is_BC_specified--
  //    Parses each face in the boundary condition section
  //    of the input file and verifies that each (variable)
  //    has a boundary conditions specified.
  //______________________________________________________________________
  void
  is_BC_specified(const ProblemSpecP& prob_spec,
                        string variable,
                  const MaterialSubset* matls)
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
    is_periodic["x-"] = periodic[0];
    is_periodic["x+"] = periodic[0];
    is_periodic["y-"] = periodic[1];
    is_periodic["y+"] = periodic[1];
    is_periodic["z-"] = periodic[2];
    is_periodic["z+"] = periodic[2];

    ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
    if(!bc_ps) {
      ostringstream warn;
      warn <<"\n__________________________________\n"
      << "ERROR: Cannot find the required xml tag"
      << "\n\t <Grid> \n \t\t<BoundaryConditions>\n \t\t</BoundaryConditions>\n\t </Grid>";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    string defaultMat="";
    ProblemSpecP defMatSpec = bc_ps->findBlock("DefaultMaterial");
    if( defMatSpec ) {
      bc_ps->get("DefaultMaterial", defaultMat);
    }

    // loop over all faces and determine if a BC has been set
    for( ProblemSpecP face_ps = bc_ps->findBlock( "Face" ); face_ps != nullptr; face_ps = face_ps->findNextBlock( "Face" ) ) {

      map<string,string> face_attr;
      face_ps->getAttributes(face_attr);

      //loop through the attributes and find  (x-,x+,y-,y+... )
      string side = "nullptr";

      for( map<string,string>::iterator iter = face_attr.begin(); iter !=  face_attr.end(); iter++ ){
        string me = (*iter).second;

        if( me =="x-" || me == "x+" ||
            me =="y-" || me == "y+" ||
            me =="z-" || me == "z+" ){
          side = me;
          is_BC_set[side] = false;
          continue;
        }
      }

      //__________________________________
      //      Loop over all BCTypes for this face:
      for( ProblemSpecP bcType_ps = face_ps->findBlock( "BCType" ); bcType_ps != nullptr; bcType_ps = bcType_ps->findNextBlock( "BCType" ) ) {
        map<string,string> bc_attr;

        bcType_ps->getAttributes(bc_attr);
        std::string label = bc_attr["label"];

        if( label != variable && label != "Symmetric" ){
          continue;
        }

        std::vector<int> matl_ids;              // find the matl_ids associated with this BC
        matl_ids = BoundCondReader::parseMatl_ids( bc_attr, defaultMat );

        for ( auto matl_id: matl_ids ){
          bool foundMatl = false;
          if( matl_id == -1 || matls->contains(matl_id)){
            foundMatl = true;
          }

          if (( label == variable || label == "Symmetric") && foundMatl ) {
            is_BC_set[side] = true;
          }
        }
      }  // BCTyoe loop

      //__________________________________
      //  Now check if the variable on this face was set
      if (!is_BC_set[side] && !is_periodic[side]){   // BC not set and not periodic
        ostringstream warn;
        warn <<"\n__________________________________\n"
        << "ERROR: The boundary condition for ( " << variable
        << " ) was not specified on face (" << side
        << ") for  materialSubset " << *matls << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }  // face loop
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
      if( periodic[2]==1 && ( is_BC_set["z-"] == true || is_BC_set["z+"] == true)){
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
  //______________________________________________________________________
  //
  void
  getBCKind( const Patch* patch,
            const Patch::FaceType face,
            const int child,
            const string& desc,
            const int mat_id,
            std::string& bc_kind,
            std::string& face_label)
  {
    bc_kind = "NotSet";

    const BoundCondBase* bc;
    const BCDataArray* bcd = patch->getBCDataArray(face);
    //__________________________________
    //  non-symmetric BCs
    // find the bc_value and kind
    //
    bc = bcd->getBoundCondData( mat_id, desc, child);

    if (bc != 0) {
      bc_kind  = bc->getBCType();
      face_label = bc->getBCFaceName();
      delete bc;
    }
  }

} // uintah namespace
