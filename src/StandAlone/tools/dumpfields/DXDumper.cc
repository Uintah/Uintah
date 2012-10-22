/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


DXDumper::DXDumper(DataArchive* da, string basedir, bool bin, bool onedim)
  : Dumper(da, basedir), nsteps_(0), dxobj_(0), bin_(bin), onedim_(onedim)
{
  // set up the file that contains a list of all the files
  this->dirname_ = this->createDirectory();
  string indexfilename = dirname_ + string("/") + string("index.dx");
  dxstrm_.open(indexfilename.c_str());
  if (!dxstrm_) {
    cerr << "Can't open output file " << indexfilename << endl;
    abort();
  }
  cout << "     " << indexfilename << endl;
}

DXDumper::~DXDumper()
{
  dxstrm_ << "object \"udadata\" class series" << endl;
  dxstrm_ << timestrm_.str();
  dxstrm_ << endl;
  dxstrm_ << "default \"udadata\"" << endl;
  dxstrm_ << "end" << endl;
  
  for(map<string,FldWriter*>::iterator fit(fldwriters_.begin());fit!=fldwriters_.end();fit++) {
    delete fit->second;
  }
}

void
DXDumper::addField(string fieldname, const Uintah::TypeDescription * /*td*/)
{
  if(fieldname=="p.particleID") return;
  fldwriters_[fieldname] = scinew FldWriter(this->dirname_, fieldname);
}

DXDumper::Step * 
DXDumper::addStep(int timestep, double time, int index)
{
  DXDumper::Step * r = scinew Step(archive(), dirname_, timestep, time, index, nsteps_++, fldwriters_, bin_, onedim_);
  return r;
}
  
void
DXDumper::finishStep(Dumper::Step * step)
{
  dxstrm_ << step->infostr() << endl;
  timestrm_ << "  member " << nsteps_-1 << " " << step->time_ << " \"stepf " << nsteps_ << "\" " << endl;
}

DXDumper::FldWriter::FldWriter(string outdir, string fieldname)
  : dxobj_(0)
{
  string outname = outdir+"/"+fieldname+".dx";
  strm_.open(outname.c_str());
  if(!strm_) {
    cerr << "Can't open output file " << outname << endl;
    abort();
  }
  cout << "     " << outname << endl;
}

DXDumper::FldWriter::~FldWriter()
{
  strm_ << "# time series " << endl;
  strm_ << "object " << ++dxobj_ << " series" << endl;
  int istep(0);
  for(list< pair<float,int> >::iterator tit(timesteps_.begin());tit!=timesteps_.end();tit++)
    {
      strm_ << "  member " << istep++ << " " << tit->first << " " << tit->second << endl;
    }
  strm_ << endl;
  
  strm_ << "default " << dxobj_ << endl;
  strm_ << "end" << endl;
}

DXDumper::Step::Step(DataArchive * da, string tsdir, int timestep, double time, int index, int fileindex, 
		     const map<string,DXDumper::FldWriter*> & fldwriters, bool bin, bool onedim)
  :
  Dumper::Step(tsdir, timestep, time, index),
  da_(da), 
  fileindex_(fileindex),
  fldwriters_(fldwriters),
  bin_(bin), onedim_(onedim)
{
  fldstrm_ << "object \"step " << fileindex_+1 << "\" class group" << endl;
}

static
double
REMOVE_SMALL(double v)
{
  if(fabs(v)<FLT_MIN) return 0;
  else return v;
}

void
DXDumper::Step::storeGrid()
{
  // FIXME: why arent I doing this (lazy ...)
}

void
DXDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  if(fieldname=="p.particleID") return;
  
  FldWriter * fldwriter = fldwriters_.find(fieldname)->second;
  ostream & os = fldwriter->strm_;
  
  GridP grid = da_->queryGrid(index_);
  
  const Uintah::TypeDescription* subtype = td->getSubType();
  
  // only support level 0 for now
  int lnum = 0;
  LevelP level = grid->getLevel(lnum);
  
  string dmode;
  if(bin_) {
    if(isLittleEndian()) dmode = " lsb";
    else                 dmode = " msb";
  } else                 dmode = " text";
  
  // build positions
  int posnobj(-1);
  int connobj(-1);
  int dataobj(-1);
  int nparts(0);
  bool iscell(false);
  IntVector nnodes, strides, minind, midind, ncells;
  if(td->getType()!=Uintah::TypeDescription::ParticleVariable) {
    IntVector indlow, indhigh;
    level->findNodeIndexRange(indlow, indhigh);
    Point x0 = level->getAnchor();
    Vector dx = level->dCell();
    
    iscell = (td->getType()==Uintah::TypeDescription::CCVariable);
    int celllen = iscell?1:0;
    nnodes  = IntVector(indhigh-indlow+IntVector(1,1,1));
    ncells  = IntVector(indhigh-indlow);
    strides = IntVector(indhigh-indlow+IntVector(1-celllen,1-celllen,1-celllen));
    minind  = indlow;
    midind  = IntVector(ncells[0]/2, ncells[1]/2, ncells[2]/2);
    
    os << "# step " << timestep_ << " positions" << endl;
    if(onedim_)
      {
	os << "object " << ++fldwriter->dxobj_ << " class array type float items " 
	   << nnodes[0] << " data follows " << endl;
	for(int i=0;i<nnodes[0];i++)
	  {
	    os << x0(0)+dx[0]*(i-minind[0]) << endl;
	  }
      }
    else
      {
	os << "object " << ++fldwriter->dxobj_ << " class gridpositions counts " 
	   << nnodes[0] << " " << nnodes[1] << " " << nnodes[2] << endl;
	os << "origin " << x0(0)-minind[0]*dx[0] << " " << x0(1)-minind[1]*dx[1] << " " << x0(2)-minind[2]*dx[2] << endl;
	os << "delta " << dx[0] << " " << 0. << " " << 0. << endl;
	os << "delta " << 0. << " " << dx[1] << " " << 0. << endl;
	os << "delta " << 0. << " " << 0. << " " << dx[2] << endl;
	os << endl;
      }
    posnobj = fldwriter->dxobj_;
    
    os << "# step " << timestep_ << " connections" << endl;
    if(onedim_)
      {
	os << "object " << ++fldwriter->dxobj_ << " class gridconnections counts " 
	   << nnodes[0] << endl; // dx wants node counts here !
	os << "attribute \"element type\" string \"lines\"" << endl;
      } 
    else
      {
	os << "object " << ++fldwriter->dxobj_ << " class gridconnections counts " 
	   << nnodes[0] << " " << nnodes[1] << " " << nnodes[2] << endl; // dx wants node counts here !
	os << "attribute \"element type\" string \"cubes\"" << endl;
      }
    os << "attribute \"ref\" string \"positions\"" << endl;
    os << endl;
    connobj = fldwriter->dxobj_;
    
    if(onedim_)
      nparts = strides(0);
    else
      nparts = strides(0)*strides(1)*strides(2);
    
  } else {
    nparts = 0;
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da_->queryMaterials("p.x", patch, index_);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	ParticleVariable<Point> partposns;
	da_->query(partposns, "p.x", matl, patch, index_);
	ParticleSubset* pset = partposns.getParticleSubset();
	nparts += pset->numParticles();
      }
    }
    
    os << "# step " << timestep_ << " positions" << endl;
    os << "object " << ++fldwriter->dxobj_ << " class array rank 1 shape 3 items " << nparts;
    os << dmode << " data follows " << endl;;
    
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da_->queryMaterials("p.x", patch, index_);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	ParticleVariable<Point> partposns;
	da_->query(partposns, "p.x", matl, patch, index_);
	ParticleSubset* pset = partposns.getParticleSubset();
	for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
	  Point xpt = partposns[*iter];
	  if(!bin_)
	    os << xpt(0) << " " << xpt(1) << " " << xpt(2) << endl;
	  else
	    {
	      for(int ic=0;ic<3;ic++) {
		float v = xpt(ic);
		os.write((char *)&v, sizeof(float));
	      }
	    }
	}
      } // materials
      
    } // patches
    os << endl;
    posnobj = fldwriter->dxobj_;
  }
  
  int ncomps, rank;
  string shp, source;
  vector<float> minval, maxval;
  
  /*if(1)*/ { // FIXME: skip p.x
    int nvals;
    switch (td->getType()) { 
    case Uintah::TypeDescription::NCVariable:
      if(onedim_)
	nvals = strides(0);
      else
	nvals = strides(0)*strides(1)*strides(2);
      source = "nodes";
      break;
    case Uintah::TypeDescription::CCVariable:
      if(onedim_)
	nvals = strides(0);
      else
	nvals = strides(0)*strides(1)*strides(2);
      source = "cells";
      break;
    case Uintah::TypeDescription::ParticleVariable:
      nvals = nparts;
      source = "particles";
      break;
    default:
      fprintf(stderr, "unexpected field type\n");
      abort();
    }
    
    cout << "     " << fieldname << endl;
    switch(subtype->getType()) {
    case Uintah::TypeDescription::float_type:  rank = 0; ncomps = 1; shp = " "; break;
    case Uintah::TypeDescription::double_type: rank = 0; ncomps = 1; shp = " "; break;
    case Uintah::TypeDescription::Point:       rank = 1; ncomps = 3; shp = "shape 3"; break;
    case Uintah::TypeDescription::Vector:      rank = 1; ncomps = 3; shp = "shape 3"; break;
    case Uintah::TypeDescription::Matrix3:     rank = 2; ncomps = 9; shp = "shape 3 3"; break;
    default: 
      fprintf(stderr, "unexpected field sub-type\n");
      abort();
    };
  
    vector<float> vals(nvals*ncomps);
    for(vector<float>::iterator vit(vals.begin());vit!=vals.end();vit++) *vit = 0.;
    
    minval.resize(ncomps);
    maxval.resize(ncomps);
    for(int ic=0;ic<ncomps;ic++) {
      minval[ic] = FLT_MAX;
      maxval[ic] =-FLT_MAX;
    }

    int ipart(0);
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da_->queryMaterials(fieldname, patch, index_);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
	  matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	switch(subtype->getType()) {
	case Uintah::TypeDescription::float_type:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<float> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		float val = REMOVE_SMALL(value[*iter]);
		vals[ipart] = val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<float> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		float val = REMOVE_SMALL(value[*iter]);
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else {
	      NCVariable<float> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		float val = REMOVE_SMALL(value[*iter]);
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    }
	  } break;
	case Uintah::TypeDescription::double_type:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<double> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		float val = REMOVE_SMALL(value[*iter]);
		vals[ipart++] = val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<double> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		IntVector ind(*iter-minind);
		cout << "index: " << ind << endl;
		if(onedim_ && (ind[1]!=midind[1] ||ind[2]!=midind[2])) continue;
		
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		double val = REMOVE_SMALL(value[*iter]);
		cout << "  val = " << val << endl;
		
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else {
	      NCVariable<double> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		float val = REMOVE_SMALL(value[*iter]);
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    }
	  } break;
	case Uintah::TypeDescription::Point:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<Point> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		for(int ic=0;ic<3;ic++) {
		  float val = REMOVE_SMALL(value[*iter](ic));
		  vals[ipart++] = val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<Point> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++){
		  float val = REMOVE_SMALL(value[*iter](ic));
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else {
	      NCVariable<Point> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++){
		  float val = REMOVE_SMALL(value[*iter](ic));
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    }
	  } break;
	case Uintah::TypeDescription::Vector:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<Vector> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		for(int ic=0;ic<3;ic++) {
		  float val = REMOVE_SMALL(value[*iter][ic]);
		  vals[ipart++] = val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<Vector> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++){
		  float val = REMOVE_SMALL(value[*iter][ic]);
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else {
	      NCVariable<Vector> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++) {
		  float val = REMOVE_SMALL(value[*iter][ic]);
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    }
	  } break;
	case Uintah::TypeDescription::Matrix3:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<Matrix3> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		for(int jc=0;jc<3;jc++)
		  for(int ic=0;ic<3;ic++) {
		    float val = REMOVE_SMALL(value[*iter](ic,jc));
		    vals[ipart++] = val;
		    if(val<minval[ic+jc*3]) minval[ic+jc*3] = val;
		    if(val>maxval[ic+jc*3]) maxval[ic+jc*3] = val;
		  }
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<Matrix3> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int jc=0;jc<3;jc++)
		  for(int ic=0;ic<3;ic++) {
		    float val = REMOVE_SMALL(value[*iter](ic,jc));
		    vals[9*ioff+ic+jc*3] += val;
		    if(val<minval[ic+jc*3]) minval[ic+jc*3] = val;
		    if(val>maxval[ic+jc*3]) maxval[ic+jc*3] = val;
		  }
	      }
	    } else {
	      NCVariable<Matrix3> value;
	      da_->query(value, fieldname, matl, patch, index_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int jc=0;jc<3;jc++)
		  for(int ic=0;ic<3;ic++) {
		    float val = REMOVE_SMALL(value[*iter](ic,jc));
		    vals[9*ioff+ic+jc*3] += val;
		    if(val<minval[ic+jc*3]) minval[ic+jc*3] = val;
		    if(val>maxval[ic+jc*3]) maxval[ic+jc*3] = val;
		  }
	      }
	    }
	  } break;
	default:
	  ;
	  // fprintf(stderr, "unexpected subtype\n");
	  // abort();
	} // subtype switch
	
      } // materials
    } // patches
    
    os << "# step " << timestep_ << " values" << endl;
    os << "object " << ++fldwriter->dxobj_ << " class array rank " << rank << " " << shp << " items " << nparts;
    os << dmode << " data follows " << endl;;
    int ioff = 0;
    for(int iv=0;iv<nvals;iv++) { 
      for(int ic=0;ic<ncomps;ic++) 
	if(!bin_)
	  os << vals[ioff++] << " ";
	else
	  os.write((char *)&vals[ioff++], sizeof(float));
      if(!bin_) os << endl;
    }
    os << endl;
    if(iscell) 
      {
	os << "attribute \"dep\" string \"connections\"" << endl;
      }
    else
      {
	os << "attribute \"dep\" string \"positions\"" << endl;
      }
    
    dataobj = fldwriter->dxobj_;
  }
  
  // build field object
  os << "# step " << timestep_ << " " << fieldname << " field" << endl;
  os << "object " << ++fldwriter->dxobj_ << " class field" << endl;
  if(posnobj!=-1) os << "  component \"positions\" value " << posnobj << endl;
  if(connobj!=-1) os << "  component \"connections\" value " << connobj << endl;
  if(dataobj!=-1) os << "  component \"data\" value " << dataobj << endl;
  os << endl;
  
  fldwriter->timesteps_.push_back( pair<float,int>(time_, fldwriter->dxobj_));
  
  int istep = fileindex_+1;
  dxstrm_ << "# step " << istep << " " << fieldname << " minimum " << endl;
  dxstrm_ << "object \"" << fieldname << " " << istep << " min\" "
	 << "class array type float rank " << rank << " " << shp << " items " << 1
	 << " data follows" << endl;
  for(int ic=0;ic<ncomps;ic++)
    dxstrm_ << minval[ic] << " ";
  dxstrm_ << endl << endl;
  
  dxstrm_ << "# step " << istep << " " << fieldname << " maximum " << endl;
  dxstrm_ << "object \"" << fieldname << " " << istep << " max\" "
	 << "class array type float rank " << rank << " " << shp << " items " << 1
	 << " data follows" << endl;
  for(int ic=0;ic<ncomps;ic++)
    dxstrm_ << maxval[ic] << " ";
  dxstrm_ << endl << endl;
  
  dxstrm_ << "object \"" << fieldname << " " << istep << " filename\" class string \"" << fieldname << ".dx\"" << endl;
  dxstrm_ << endl;
  
  dxstrm_ << "# step " << istep << " info " << endl;
  dxstrm_ << "object \"" << fieldname << " " << istep << " info\" class group" << endl;
  dxstrm_ << "  member \"minimum\" \"" << fieldname << " " << istep << " min\"" << endl;
  dxstrm_ << "  member \"maximum\" \"" << fieldname << " " << istep << " max\"" << endl;
  dxstrm_ << "  member \"filename\" \"" << fieldname << " " << istep << " filename\"" << endl;
  dxstrm_ << "  attribute \"source\" string \"" << source << "\"" << endl;
  dxstrm_ << endl;
  
  fldstrm_ << "  member \"" << fieldname <<  "\" " << "\"" << fieldname << " " << istep << " info\"" << endl;
}

