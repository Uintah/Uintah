/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/MPMBoundCond.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

using namespace Uintah;
using namespace std;

static DebugStream dbg_BC("MPM_BC", false);

MPMBoundCond::MPMBoundCond()
{
}

MPMBoundCond::~MPMBoundCond()
{
}
//______________________________________________________________________
//
void MPMBoundCond::setBoundaryCondition(const Patch* patch,
                                        int dwi,
                                        const string& type, 
                                        NCVariable<Vector>& variable,
                                        string interp_type)
{
  dbg_BC << "-------- setBC(NC_Vector)  \t"<< type <<" "
          << " mat_id = " << dwi <<  ", Patch: "<< patch->getID() << endl;
          
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
      
    IntVector oneCell = patch->faceDirection(face);

    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      
      IntVector l(0,0,0),h(0,0,0),off(0,0,0);
      
      if(interp_type=="gimp" || interp_type=="cpdi" ||
         interp_type=="cpti" || interp_type=="3rdorderBS" ||
         interp_type=="fast_cpdi"){
        patch->getFaceExtraNodes(face,0,l,h);
      }
      
      for (int child = 0; child < numChildren; child++) {
        Iterator nbound_ptr;
        Iterator nu;        // not used;

        if (type == "Velocity"){
         const  BoundCondBase* bcb = 
            patch->getArrayBCValues(face,dwi,"Velocity",nu,nbound_ptr,child);
          
          if (!bcb){
            continue;
          }
          
         const BoundCond<Vector>* bc = 
           dynamic_cast<const BoundCond<Vector>*>(bcb); 
           
          if (bc != 0) {
            if (bc->getBCType() == "Dirichlet") {
              Vector bcv = bc->getValue();
              for (nbound_ptr.reset();!nbound_ptr.done();nbound_ptr++){ 
                IntVector nd = *nbound_ptr;
                variable[nd] = bcv;
              }
              if(interp_type=="gimp" || interp_type=="3rdorderBS" ||
                 interp_type=="cpdi" || interp_type=="cpti" ||
                 interp_type=="fast_cpdi"){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = bcv;
                }
              }
            }
            //__________________________________
            //  debugging
            if( dbg_BC.active() ) {
              nbound_ptr.reset();
              dbg_BC <<"Face: "<< patch->getFaceName(face) << " interp: " << interp_type
                   <<"\t child " << child  <<" NumChildren "<<numChildren
                   <<"\t bound_ptr = "<< nbound_ptr<< endl;
            }
            
            delete bc;
          } else
          delete bcb;

        } else if (type == "Symmetric"){
          const BoundCondBase* bcb =
            patch->getArrayBCValues(face,dwi,"Symmetric",nu,nbound_ptr,child);
          if (!bcb) { 
            continue;
          }
          
          if (bcb->getBCType() == "symmetry") {
            
            //__________________________________
            //
            if (face == Patch::xplus || face == Patch::xminus){
              if(interp_type=="linear"){
               for (nbound_ptr.reset(); !nbound_ptr.done();nbound_ptr++) {
                IntVector nd = *nbound_ptr;
                variable[nd] = Vector(0.,variable[nd].y(), variable[nd].z());
               }
              } // linear
      
              if(interp_type=="gimp" || interp_type=="cpdi" ||
                 interp_type=="cpti" || interp_type=="3rdorderBS" ||
                 interp_type=="fast_cpdi"){
                IntVector off = IntVector(1,0,0);
                IntVector L(0,0,0),H(0,0,0);
                IntVector inner;
                if(face==Patch::xminus){
                  L = l+off; H = h+off;
                  for(NodeIterator it(L,H); !it.done(); it++){//bndy face nodes
                    IntVector nd = *it;
                    variable[nd]=Vector(0.,variable[nd].y(), variable[nd].z());
                  }
                } else if(face==Patch::xplus){
                  L = l-off; H = h-off;
                  for(NodeIterator it(L,H); !it.done(); it++){//bndy face nodes
                    IntVector nd = *it;
                    variable[nd]=Vector(0.,variable[nd].y(), variable[nd].z());
                  }
                }
                if(face==Patch::xminus){
                  inner = IntVector(2,0,0);
                  for(NodeIterator it(l,h); !it.done(); it++) { //extra nodes
                    IntVector nd = *it;
                    variable[nd] = Vector(-variable[nd+inner].x(),
                                           variable[nd+inner].y(), 
                                           variable[nd+inner].z());
                  }
                } else if(face==Patch::xplus){
                  inner = IntVector(-2,0,0);
                  for(NodeIterator it(l,h); !it.done(); it++) { //extra nodes
                    IntVector nd = *it;
                    variable[nd] = Vector(-variable[nd+inner].x(),
                                           variable[nd+inner].y(), 
                                           variable[nd+inner].z());
                  }
                }
              }  // cpdi, cpti, gimp or 3rdorderBS
            } // xplus/xminus faces
            
            //__________________________________
            //
            if (face == Patch::yplus || face == Patch::yminus){
              if(interp_type=="linear"){
               for (nbound_ptr.reset(); !nbound_ptr.done();nbound_ptr++){
                IntVector nd = *nbound_ptr;
                variable[nd] = Vector(variable[nd].x(),0.,variable[nd].z());
               }
              } // linear
      
              if(interp_type=="gimp" || interp_type=="cpdi" ||
                 interp_type=="cpti" || interp_type=="3rdorderBS" ||
                 interp_type=="fast_cpdi"){
                IntVector off = IntVector(0,1,0);
                IntVector L(0,0,0),H(0,0,0);
                IntVector inner;
                if(face==Patch::yminus){
                  L = l+off; H = h+off;
                  for(NodeIterator it(L,H); !it.done(); it++){//bndy face nodes
                    IntVector nd = *it;
                    variable[nd]=Vector(variable[nd].x(),0.,variable[nd].z());
                  }
                } else if(face==Patch::yplus){
                  L = l-off; H = h-off;
                  for(NodeIterator it(L,H); !it.done(); it++){//bndy face nodes
                    IntVector nd = *it;
                    variable[nd]=Vector(variable[nd].x(),0.,variable[nd].z());
                  }
                }
                if(face==Patch::yminus){
                  inner = IntVector(0,2,0);
                  for(NodeIterator it(l,h); !it.done(); it++) { // extra nodes
                    IntVector nd = *it;
                    variable[nd] = Vector(variable[nd+inner].x(),
                                         -variable[nd+inner].y(),
                                          variable[nd+inner].z());
                  }
                } else if(face==Patch::yplus){
                  inner = IntVector(0,-2,0);
                  for(NodeIterator it(l,h); !it.done(); it++) { // extra nodes
                    IntVector nd = *it;
                    variable[nd] = Vector(variable[nd+inner].x(),
                                         -variable[nd+inner].y(),
                                          variable[nd+inner].z());
                  }
                }
              } // cpdi, cpti or gimp
            }  // yplus/yminus faces
            //__________________________________
            //
            if (face == Patch::zplus || face == Patch::zminus){
              if(interp_type=="linear"){
              
               for (nbound_ptr.reset(); !nbound_ptr.done();nbound_ptr++){
                IntVector nd = *nbound_ptr;
                variable[nd] = Vector(variable[nd].x(), variable[nd].y(),0.);
               }
              } // linear
              
              if(interp_type=="gimp" || interp_type=="cpdi" ||
                 interp_type=="cpti" || interp_type=="3rdorderBS" ||
                 interp_type=="fast_cpdi"){
                IntVector off = IntVector(0,0,1);
                IntVector L(0,0,0),H(0,0,0);
                IntVector inner;
                if(face==Patch::zminus){
                  L = l+off; H = h+off;
                  for(NodeIterator it(L,H); !it.done(); it++){//bndy face nodes
                    IntVector nd = *it;
                    variable[nd]=Vector(variable[nd].x(), variable[nd].y(),0.);
                  }
                } else if(face==Patch::zplus){
                  L = l-off; H = h-off;
                  for(NodeIterator it(L,H); !it.done(); it++){//bndy face nodes
                    IntVector nd = *it;
                    variable[nd]=Vector(variable[nd].x(), variable[nd].y(),0.);
                  }
                }
                if(l.z()==-1 || h.z()==3){
                 if(face==Patch::zminus){
                  inner = IntVector(0,0,2);
                  for(NodeIterator it(l,h); !it.done(); it++) { // extra nodes
                    IntVector nd = *it;
                    variable[nd] = Vector(variable[nd+inner].x(),
                                          variable[nd+inner].y(),
                                         -variable[nd+inner].z());
                  }
                 } else if(face==Patch::zplus){
                  inner = IntVector(0,0,-2);
                  for(NodeIterator it(l,h); !it.done(); it++) { // extra nodes
                    IntVector nd = *it;
                    variable[nd] = Vector(variable[nd+inner].x(),
                                          variable[nd+inner].y(),
                                         -variable[nd+inner].z());
                  }
                 }
                }
              } // cpdi, cpti or gimp
            } // zplus/zminus
            
            //__________________________________
            //  debugging
            if( dbg_BC.active() ) {
              nbound_ptr.reset();
              dbg_BC <<"Face: "<< patch->getFaceName(face) << " interp: " << interp_type
                   <<"\t child " << child  <<" NumChildren "<<numChildren
                   <<"\t bound_ptr = "<< nbound_ptr<< endl;
            }
            delete bcb;
          } else{
            delete bcb;
          }
        }
      }
    } else
      continue;
  }  // face loop
}

//______________________________________________________________________
//
void MPMBoundCond::setBoundaryCondition(const Patch* patch,
                                        int dwi,
                                        const string& type, 
                                        NCVariable<double>& variable,
                                        string interp_type)
{

  dbg_BC << "-------- setBC(NC_double)  \t"<< type << " " 
            << " mat_id = " << dwi <<  ", Patch: "<< patch->getID() << endl;
            
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
      
    IntVector oneCell = patch->faceDirection(face);
    
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      IntVector l(0,0,0),h(0,0,0);
      
      if(interp_type=="gimp" || interp_type=="cpdi" ||
         interp_type=="cpti" || interp_type=="3rdorderBS" ||
         interp_type=="fast_cpdi"){
        patch->getFaceExtraNodes(face,0,l,h);
      }
      
      for (int child = 0; child < numChildren; child++) {
        Iterator nbound_ptr;
        Iterator nu;  // not used


        //__________________________________
        // Used in MPMICE and Scalar diffusion
        if(type=="Pressure" || type=="Temperature" || type=="SD-Type"){
          const BoundCondBase *bcb = 
            patch->getArrayBCValues(face,dwi,type,nu,nbound_ptr, child);
            
          if (!bcb){
            continue;
          }
          const BoundCond<double>* bc = 
            dynamic_cast<const BoundCond<double>*>(bcb);
          
          if (bc != 0) {
            //__________________________________
            //
            if (bc->getBCType() == "Dirichlet") {
              double bcv = bc->getValue();
              
              for (nbound_ptr.reset(); !nbound_ptr.done();nbound_ptr++){
                IntVector nd = *nbound_ptr;
                variable[nd] = bcv;
              }
              
              if(interp_type=="gimp" || interp_type=="cpdi" ||
                 interp_type=="cpti" || interp_type=="3rdorderBS" ||
                 interp_type=="fast_cpdi"){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = bcv;
                }
              }
            }
            
            //__________________________________
            //
            if (bc->getBCType() == "Neumann"){   // There's nothing to do for symmetric BCs - Jim
              Vector deltax = patch->dCell();
              double dx = -9;
              IntVector off(-9,-9,-9);
              if (face == Patch::xplus){
                dx = deltax.x();
                off=IntVector(1,0,0);
              }
              else if (face == Patch::xminus){
                dx = deltax.x();
                off=IntVector(-1,0,0);
              }
              else if (face == Patch::yplus){
                dx = deltax.y();
                off=IntVector(0,1,0);
              }
              else if (face == Patch::yminus){
                dx = deltax.y();
                off=IntVector(0,-1,0);
              }
              else if (face == Patch::zplus){
                dx = deltax.z();
                off=IntVector(0,0,1);
              }
              else if (face == Patch::zminus){
                dx = deltax.z();
                off=IntVector(0,0,-1);
              }
              
              double gradv = bc->getValue();

              for (nbound_ptr.reset(); !nbound_ptr.done(); nbound_ptr++) {
		IntVector nd = *nbound_ptr;
		variable[nd] = variable[nd-off] - gradv*dx;
	      }

              for(NodeIterator it(l,h); !it.done(); it++) {
                IntVector nd = *it;
                variable[nd] = variable[nd-off] - gradv*dx;
              }
            }
            
            //__________________________________
            //  debugging
            if( dbg_BC.active() ) {
              nbound_ptr.reset();
              dbg_BC <<"Face: "<< patch->getFaceName(face) << " interp: " << interp_type
                     <<"\t child " << child  <<" NumChildren "<<numChildren
                     <<"\t bound_ptr = "<< nbound_ptr<< endl;
            }
            
            delete bc;
          } else
          delete bcb;
        }
      }  // child
    } else
      continue;
  }
}
