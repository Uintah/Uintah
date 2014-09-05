/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 * GLTextureBuilder.cc
 *
 * Simple interface to volume rendering stuff
 */


#include <Core/Containers/Array1.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Datatypes/VolumeUtils.h>
#include <Dataflow/Network/Module.h>

#include <sys/types.h>
#include <unistd.h>

#include <Packages/Uintah/Core/Datatypes/GLTexture3D.h>
#include <Dataflow/Modules/Visualization/GLTextureBuilder.h>

#include <iostream>
using std::cerr;
using std::endl;


namespace Uintah {

class GLTextureBuilder : public SCIRun::GLTextureBuilder {

public:
  GLTextureBuilder( const string& id);

  virtual ~GLTextureBuilder();
  virtual void execute();

};


extern "C" Module* make_GLTextureBuilder( const string& id) {
  return scinew GLTextureBuilder(id);
}


GLTextureBuilder::GLTextureBuilder(const string& id)
  : SCIRun::GLTextureBuilder( id )
{
  packageName = "Uintah";
}

GLTextureBuilder::~GLTextureBuilder()
{
}

void GLTextureBuilder::execute(void)
{

  FieldHandle sfield;
  infield_ = (FieldIPort *)get_iport("Field");
  otexture_ = (GLTexture3DOPort *)get_oport("GL Texture");
  if (!infield_->get(sfield)){
    return;
  } else if( sfield.get_rep() && sfield->get_type_name(0) == "LatticeVol"){
    SCIRun::GLTextureBuilder::real_execute(sfield);
    return;
  }
  else if (!sfield.get_rep() || sfield->get_type_name(0) != "LevelField") {
    return;
  }
  
  reset_vars();
  double minV, maxV;
  double min = min_.get();
  double max = max_.get();
  int is_fixed = is_fixed_.get();
  cerr << "is_fixed = "<<is_fixed<<"\n";
  if (sfield.get_rep() != sfrg_.get_rep()  && !tex_.get_rep())
  {
    sfrg_ = sfield;
    if (is_fixed) {  // if fixed, copy min/max into these locals
      minV = min;
      maxV = max;
    }

    // this constructor will take in a min and max, and if is_fixed is set
    // it will set the values to that range... otherwise it auto-scales
    tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);

    if(!is_fixed){
      min_.set(minV);
      max_.set(maxV);
    }

    TCL::execute(id + " SetDims " + to_string( tex_->get_brick_size()));
    max_brick_dim_.set(tex_->get_brick_size());
    old_brick_size_ = tex_->get_brick_size();
  }
  else if (sfield.get_rep() != sfrg_.get_rep())
  {
    // see note above
    sfrg_ = sfield;
    if (is_fixed) {  // if fixed, copy min/max into these locals
      minV = min;
      maxV = max;
    }
    tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);

    if(!is_fixed){
      min_.set(minV);
      max_.set(maxV);
    }
    tex_->set_brick_size(max_brick_dim_.get());
  }
  else if (old_brick_size_ != max_brick_dim_.get())
  {
    tex_->set_brick_size(max_brick_dim_.get());
    old_brick_size_ = max_brick_dim_.get();
  }
  else if ((old_min_ != min) || (old_max_ != max))
  {
    if (is_fixed) {
      minV = min;
      maxV = max;
    }

    // see note above
    tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);
    if(!is_fixed){
      min_.set(minV);
      max_.set(maxV);
    }
 }    

  old_min_ = (int)minV;
  old_max_ = (int)maxV;

  if (tex_.get_rep())
  {
    otexture_->send(tex_);
  }
}

} // End namespace Uintah


