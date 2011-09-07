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



#include <StandAlone/tools/uda2nrrd/update_mesh_handle.h>

#include <StandAlone/tools/uda2nrrd/bc.h>
#include <StandAlone/tools/uda2nrrd/handleVariable.h>

bool 
update_mesh_handle( LevelP& level,
                    IntVector& hi,
                    IntVector& range,
                    BBox& box,
                    Uintah::TypeDescription::Type type,
                    LVMeshHandle& mesh_handle,
                    const Args & args )
{
  //   cerr<<"In update_mesh_handled: type = "<<type<<"\n";
  
  switch ( type ){
  case Uintah::TypeDescription::CCVariable:
    {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( args.remove_boundary ){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y(),
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y(),
                                          range.z(), box.min(),
                                          box.max());
//                cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                  range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
                mesh_handle->get_nj() != (unsigned int) range.y() ||
                mesh_handle->get_nk() != (unsigned int) range.z() ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y(),
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y(),
                                          range.z(), box.min(),
                                          box.max());
//                cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                  range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
        }
      } 
      return true;
    }
  case Uintah::TypeDescription::NCVariable:
    {
      if( mesh_handle.get_rep() == 0 ){
        mesh_handle = scinew LVMesh(range.x(), range.y(),
                                        range.z(), box.min(),
                                        box.max());
//              cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
      }else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
               mesh_handle->get_nj() != (unsigned int) range.y() ||
               mesh_handle->get_nk() != (unsigned int) range.z() ){
        mesh_handle = scinew LVMesh(range.x(), range.y(),
                                        range.z(), box.min(),
                                        box.max());
//              cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
      }
      return true;
    }
  case Uintah::TypeDescription::SFCXVariable:
    {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( args.remove_boundary ){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
//           cerr<<"is periodic?\n";
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y() - 1,
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y() - 1,
                                      range.z() - 1, box.min(),
                                      box.max());
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
                mesh_handle->get_nj() != (unsigned int) range.y() -1 ||
                mesh_handle->get_nk() != (unsigned int) range.z() -1 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y() - 1,
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y() - 1,
                                      range.z()-1, box.min(),
                                      box.max());
        }
      }
      return true;
    }
  case Uintah::TypeDescription::SFCYVariable:
    {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( args.remove_boundary ){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y(),
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y(),
                                      range.z()-1, box.min(),
                                      box.max());
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() -1 ||
                mesh_handle->get_nj() != (unsigned int) range.y() ||
                mesh_handle->get_nk() != (unsigned int) range.z() -1 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y(),
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y(),
                                      range.z()-1, box.min(),
                                      box.max());
        }
      }
      return true;
    }
  case Uintah::TypeDescription::SFCZVariable:
     {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( args.remove_boundary ){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y() - 1,
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y()-1,
                                      range.z(), box.min(),
                                      box.max());
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() -1 ||
                mesh_handle->get_nj() != (unsigned int) range.y() -1 ||
                mesh_handle->get_nk() != (unsigned int) range.z() ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y() - 1,
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y()-1,
                                      range.z(), box.min(),
                                      box.max());
        }
      }     
      return true;
    }
  default:
    return false;
  }
} // end update_mesh_handled()

