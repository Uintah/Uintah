
#ifndef ASETOKENS_H
#define ASETOKENS_H 1



#include <Packages/rtrt/Core/Token.h>
#include <string>
#include <stdlib.h>
#ifdef _WIN32
#include <float.h>
#else
#include <values.h>
#endif
#include <sys/stat.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>



class SceneToken : public Token
{

 public:

  SceneToken() : Token("*SCENE") {
    AddChildMoniker("*SCENE_ENVMAP");
    AddChildMoniker("*SCENE_AMBIENT_STATIC");
  }
  virtual ~SceneToken() {}
    
  virtual Token *MakeToken() { return new SceneToken(); }
};



class SceneEnvMapToken : public Token
{

 public:
  
  SceneEnvMapToken() : Token("*SCENE_ENVMAP") {
    AddChildMoniker("*BITMAP");
  }
  virtual ~SceneEnvMapToken() {}

  virtual Token *MakeToken() { return new SceneEnvMapToken(); }
};



class SceneAmbientStaticToken : public Token
{

 public:

  SceneAmbientStaticToken() : Token("*SCENE_AMBIENT_STATIC") { nargs_ = 3; }
  virtual ~SceneAmbientStaticToken() {}

  virtual Token *MakeToken() { return new SceneAmbientStaticToken(); }
};



class MaterialListToken : public Token 
{

 public:

  MaterialListToken() : Token("*MATERIAL_LIST") {
    AddChildMoniker("*MATERIAL");
  }
  virtual ~MaterialListToken() {}

  virtual Token *MakeToken() { return new MaterialListToken(); }
};



class BitmapToken : public Token
{

 public:

  BitmapToken() : Token("*BITMAP") { nargs_ = 1; }
  virtual ~BitmapToken() {}

  virtual void Write(ofstream &str) {
    Indent(str);
    str << moniker_ << " \"" << args_[0] << "\"" << endl;
  }

  virtual Token *MakeToken() { return new BitmapToken(); }
};



class MaterialToken : public Token
{

 protected:

  string   name_;
  unsigned index_;
  double   ambient_[3];
  double   diffuse_[3];
  double   specular_[3];
  double   shine_;
  double   transparency_;
  string   tmap_filename_;
  string   bmap_filename_;
  string   omap_filename_;
  string   imap_filename_;

 public:

  MaterialToken() : Token("*MATERIAL") {
    nargs_ = 1;
    AddChildMoniker("*MATERIAL_NAME");
    AddChildMoniker("*MATERIAL_AMBIENT");
    AddChildMoniker("*MATERIAL_DIFFUSE");
    AddChildMoniker("*MATERIAL_SPECULAR");
    AddChildMoniker("*MATERIAL_SHINE");
    AddChildMoniker("*MATERIAL_TRANSPARENCY");
    AddChildMoniker("*MAP_DIFFUSE");
    AddChildMoniker("*MAP_BUMP");
    AddChildMoniker("*MAP_OPACITY");
    AddChildMoniker("*MAP_SELFILLUM");
    //AddChildMoniker("*SUBMATERIAL");
  }
  virtual ~MaterialToken() {}

  virtual bool Parse(ifstream &str) {
    str >> index_;
    ParseChildren(str);

    cout << "Material " << index_ << ": " << name_ << endl
         << ambient_[0] << ", " << ambient_[1] << ", " << ambient_[2] << endl
         << diffuse_[0] << ", " << diffuse_[1] << ", " << diffuse_[2] << endl
         << specular_[0] << ", " << specular_[1] << ", " << specular_[2] 
         << endl
         << shine_ << endl
         << transparency_ << endl 
         << tmap_filename_ << endl << endl;
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MATERIAL " << index_ << " {" << endl;
    ++indent_;
    unsigned length = children_.size();
    for (unsigned loop=0; loop<length; ++loop)
      children_[loop]->Write(str);
    --indent_;
    Indent(str);
    str << "}" << endl;
  }

  unsigned GetIndex() { return index_; }
  
  string GetName() { return name_; }
  void SetName(const string& s) { name_ = s; }
  
  void GetAmbient(double c[3]) { 
    c[0] = ambient_[0];
    c[1] = ambient_[1];
    c[2] = ambient_[2];
  } 
  void SetAmbient(double c[3]) { 
    ambient_[0] = c[0];
    ambient_[1] = c[1];
    ambient_[2] = c[2];
  } 

  void GetDiffuse(double c[3]) { 
    c[0] = diffuse_[0];
    c[1] = diffuse_[1];
    c[2] = diffuse_[2];
  } 
  void SetDiffuse(double c[3]) { 
    diffuse_[0] = c[0];
    diffuse_[1] = c[1];
    diffuse_[2] = c[2];
  } 

  void GetSpecular(double c[3]) { 
    c[0] = specular_[0];
    c[1] = specular_[1];
    c[2] = specular_[2];
  } 
  void SetSpecular(double c[3]) { 
    specular_[0] = c[0];
    specular_[1] = c[1];
    specular_[2] = c[2];
  } 

  double GetShine() { return shine_; } 
  void SetShine(double s) { shine_ = s; } 

  double GetTransparency() { return transparency_; }
  void SetTransparency(double s) { transparency_ = s; }

  string GetTMapFilename() { return tmap_filename_; }
  void SetTMapFilename(const string& s) { tmap_filename_ = s; }

  string GetBMapFilename() { return bmap_filename_; }
  void SetBMapFilename(const string& s) { bmap_filename_ = s; }

  string GetOMapFilename() { return omap_filename_; }
  void SetOMapFilename(const string& s) { omap_filename_ = s; }

  string GetIMapFilename() { return imap_filename_; }
  void SetIMapFilename(const string& s) { imap_filename_ = s; }

  Token *MakeToken() { return new MaterialToken(); }
};



class MaterialNameToken : public Token
{

 public:
  
  MaterialNameToken() : Token("*MATERIAL_NAME") { nargs_ = 1; }
  ~MaterialNameToken() {}

  virtual bool Parse(ifstream &str) {
    ParseArgs(str);
    ((MaterialToken*)parent_)->SetName(args_[0]);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MATERIAL_NAME \"" << ((MaterialToken*)parent_)->GetName() 
        << "\"" << endl;
  }

  Token *MakeToken() { return new MaterialNameToken(); }
};



class MaterialAmbientToken : public Token
{

 public:
  
  MaterialAmbientToken() : Token("*MATERIAL_AMBIENT") {}
  ~MaterialAmbientToken() {}

  virtual bool Parse(ifstream &str) {
    double color[3];
    str >> color[0] >> color[1] >> color [2];
    ((MaterialToken*)parent_)->SetAmbient(color);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    double color[3];
    ((MaterialToken*)parent_)->GetAmbient(color);
    str << "*MATERIAL_AMBIENT " 
        << color[0] << " " 
        << color[1] << " " 
        << color[2] << endl;
  }

  Token *MakeToken() { return new MaterialAmbientToken(); }
};



class MaterialDiffuseToken : public Token
{

 public:
  
  MaterialDiffuseToken() : Token("*MATERIAL_DIFFUSE") {}
  ~MaterialDiffuseToken() {}

  virtual bool Parse(ifstream &str) {
    double color[3];
    str >> color[0] >> color[1] >> color [2];
    ((MaterialToken*)parent_)->SetDiffuse(color);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    double color[3];
    ((MaterialToken*)parent_)->GetDiffuse(color);
    str << "*MATERIAL_DIFFUSE " 
        << color[0] << " " 
        << color[1] << " " 
        << color[2] << endl;
  }

  Token *MakeToken() { return new MaterialDiffuseToken(); }
};



class MaterialSpecularToken : public Token
{

 public:
  
  MaterialSpecularToken() : Token("*MATERIAL_SPECULAR") {}
  ~MaterialSpecularToken() {}

  virtual bool Parse(ifstream &str) {
    double color[3];
    str >> color[0] >> color[1] >> color [2];
    ((MaterialToken*)parent_)->SetSpecular(color);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    double color[3];
    ((MaterialToken*)parent_)->GetSpecular(color);
    str << "*MATERIAL_SPECULAR " 
        << color[0] << " " 
        << color[1] << " " 
        << color[2] << endl;
  }

  Token *MakeToken() { return new MaterialSpecularToken(); }
};



class MaterialShineToken : public Token
{

 public:
  
  MaterialShineToken() : Token("*MATERIAL_SHINE") {}
  ~MaterialShineToken() {}

  virtual bool Parse(ifstream &str) {
    double shine;
    str >> shine;
    ((MaterialToken*)parent_)->SetShine(shine);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MATERIAL_SHINE " << ((MaterialToken*)parent_)->GetShine() << endl;
  }

  Token *MakeToken() { return new MaterialShineToken(); }
};



class MaterialTransparencyToken : public Token
{

 public:
  
  MaterialTransparencyToken() : Token("*MATERIAL_TRANSPARENCY") {}
  ~MaterialTransparencyToken() {}

  virtual bool Parse(ifstream &str) {
    double t;
    str >> t;
    ((MaterialToken*)parent_)->SetTransparency(t);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MATERIAL_TRANSPARENCY " 
        << ((MaterialToken*)parent_)->GetTransparency() << endl;
  }

  Token *MakeToken() { return new MaterialTransparencyToken(); }
};



class MapDiffuseToken : public Token
{

 public:
  
  MapDiffuseToken() : Token("*MAP_DIFFUSE") {
    AddChildMoniker("*BITMAP");
  }
  ~MapDiffuseToken() {}

  virtual bool Parse(ifstream &str) {
    ParseChildren(str);
    if (children_.size()>0) {
      string name = (*(children_[0]->GetArgs()))[0];
      ((MaterialToken*)parent_)->SetTMapFilename(name);
    }
    return true;
  }

  Token *MakeToken() { return new MapDiffuseToken(); }
};



class MapBumpToken : public Token
{

 public:
  
  MapBumpToken() : Token("*MAP_BUMP") {
    AddChildMoniker("*BITMAP");
  }
  ~MapBumpToken() {}

  virtual bool Parse(ifstream &str) {
    ParseChildren(str);
    if (children_.size()>0) {
      string name = (*(children_[0]->GetArgs()))[0];
      ((MaterialToken*)parent_)->SetBMapFilename(name);
    }
    return true;
  }

  Token *MakeToken() { return new MapBumpToken(); }
};



class MapOpacityToken : public Token
{

 public:
  
  MapOpacityToken() : Token("*MAP_OPACITY") { 
    AddChildMoniker("*BITMAP");
  }
  ~MapOpacityToken() {}

  virtual bool Parse(ifstream &str) {
    ParseChildren(str);
    if (children_.size()>0) {
      string name = (*(children_[0]->GetArgs()))[0];
      ((MaterialToken*)parent_)->SetOMapFilename(name);
    }
    return true;
  }

  Token *MakeToken() { return new MapOpacityToken(); }
};



class MapSelfIllumToken : public Token
{

 public:
  
  MapSelfIllumToken() : Token("*MAP_SELFILLUM") { 
    AddChildMoniker("*BITMAP");
  }
  ~MapSelfIllumToken() {}

  virtual bool Parse(ifstream &str) {
    ParseChildren(str);
    if (children_.size()>0) {
      string name = (*(children_[0]->GetArgs()))[0];
      ((MaterialToken*)parent_)->SetIMapFilename(name);
    }
    return true;
  }

  Token *MakeToken() { return new MapSelfIllumToken(); }
};



class SubMaterialToken : public Token
{

 protected:

  string   name_;
  unsigned index_;
  double   ambient_[3];
  double   diffuse_[3];
  double   specular_[3];
  double   shine_;
  double   transparency_;
  string   tmap_filename_;
  string   bmap_filename_;
  string   omap_filename_;
  string   imap_filename_;

 public:

  SubMaterialToken() : Token("*MATERIAL") {
    nargs_ = 1;
    AddChildMoniker("*MATERIAL_NAME");
    AddChildMoniker("*MATERIAL_AMBIENT");
    AddChildMoniker("*MATERIAL_DIFFUSE");
    AddChildMoniker("*MATERIAL_SPECULAR");
    AddChildMoniker("*MATERIAL_SHINE");
    AddChildMoniker("*MATERIAL_TRANSPARENCY");
    AddChildMoniker("*MAP_DIFFUSE");
    AddChildMoniker("*MAP_BUMP");
    AddChildMoniker("*MAP_OPACITY");
    AddChildMoniker("*MAP_SELFILLUM");
  }
  virtual ~SubMaterialToken() {}

  virtual bool Parse(ifstream &str) {
    str >> index_;
    ParseChildren(str);

    cout << "SubMaterial " << index_ << ": " << name_ << endl
         << ambient_[0] << ", " << ambient_[1] << ", " << ambient_[2] << endl
         << diffuse_[0] << ", " << diffuse_[1] << ", " << diffuse_[2] << endl
         << specular_[0] << ", " << specular_[1] << ", " << specular_[2] 
         << endl
         << shine_ << endl
         << transparency_ << endl 
         << tmap_filename_ << endl 
         << bmap_filename_ << endl 
         << omap_filename_ << endl 
         << imap_filename_ << endl << endl;
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*SUBMATERIAL " << index_ << " {" << endl;
    ++indent_;
    unsigned length = children_.size();
    for (unsigned loop=0; loop<length; ++loop)
      children_[loop]->Write(str);
    --indent_;
    Indent(str);
    str << "}" << endl;
  }

  unsigned GetIndex() { return index_; }

  string GetName() { return name_; }
  void SetName(const string& s) { name_ = s; }
  
  void GetAmbient(double c[3]) { 
    c[0] = ambient_[0];
    c[1] = ambient_[1];
    c[2] = ambient_[2];
  } 
  void SetAmbient(double c[3]) { 
    ambient_[0] = c[0];
    ambient_[1] = c[1];
    ambient_[2] = c[2];
  } 

  void GetDiffuse(double c[3]) { 
    c[0] = diffuse_[0];
    c[1] = diffuse_[1];
    c[2] = diffuse_[2];
  } 
  void SetDiffuse(double c[3]) { 
    diffuse_[0] = c[0];
    diffuse_[1] = c[1];
    diffuse_[2] = c[2];
  } 

  void GetSpecular(double c[3]) { 
    c[0] = specular_[0];
    c[1] = specular_[1];
    c[2] = specular_[2];
  } 
  void SetSpecular(double c[3]) { 
    specular_[0] = c[0];
    specular_[1] = c[1];
    specular_[2] = c[2];
  } 

  double GetShine() { return shine_; } 
  void SetShine(double s) { shine_ = s; } 

  double GetTransparency() { return transparency_; }
  void SetTransparency(double s) { transparency_ = s; }

  string GetTMapFilename() { return tmap_filename_; }
  void SetTMapFilename(const string& s) { tmap_filename_ = s; }

  string GetBMapFilename() { return bmap_filename_; }
  void SetBMapFilename(const string& s) { bmap_filename_ = s; }

  string GetOMapFilename() { return omap_filename_; }
  void SetOMapFilename(const string& s) { omap_filename_ = s; }

  string GetIMapFilename() { return imap_filename_; }
  void SetIMapFilename(const string& s) { imap_filename_ = s; }

  Token *MakeToken() { return new SubMaterialToken(); }
};



class GeomObjectToken : public Token
{

 protected:

  bool     empty_;
  string   nodename_;
  unsigned material_index_;

 public:

  GeomObjectToken() : Token("*GEOMOBJECT"), empty_(false), material_index_(0) {
    AddChildMoniker("*MATERIAL_REF");
    AddChildMoniker("*NODE_NAME");
    AddChildMoniker("*MESH");
  }
  virtual ~GeomObjectToken() {}

  virtual bool Parse(ifstream &str) {
    ParseArgs(str);
    cout << endl << "Token: *GEOMOBJECT" << endl;
    ParseChildren(str);
    return true;
  }

  void SetEmpty() { empty_ = true; }
  bool Empty() { return empty_; }
  
  void SetNodeName(const string& s) { nodename_ = s; }
  string GetNodeName() { return nodename_; }

  void SetMaterialIndex(unsigned i) { 
    material_index_ = i; 
  }
  unsigned GetMaterialIndex() { return material_index_; }

  virtual Token *MakeToken() { return new GeomObjectToken(); }
};



class NodeNameToken : public Token
{

 public:

  NodeNameToken() : Token("*NODE_NAME") { nargs_ = 1; }
  virtual ~NodeNameToken() {}

  virtual bool Parse(ifstream &str) {
    bool status = ParseArgs(str);
    if (status) 
      cout << "Token: *NODE_NAME = " << args_[0] << endl;

    if (parent_->GetMoniker() == "*GEOMOBJECT") {
      ((GeomObjectToken*)parent_)->SetNodeName(args_[0]);
    }

    return status;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << moniker_ << " \"" << args_[0] << "\"" << endl;
  }

  virtual Token *MakeToken() { return new NodeNameToken(); }
};



class MeshToken : public Token
{

  unsigned numvertices_;
  unsigned numfaces_;

  unsigned numtvertices_;
  unsigned numtfaces_;

  vector<double>   *vertices_;
  vector<double>   *tvertices_;
  vector<unsigned> *faces_;
  vector<unsigned> *tfaces_;
  vector<double>   *face_normals_;
  vector<double>   *vertex_normals_;

 public:

  MeshToken() : Token("*MESH"), numvertices_(0), numfaces_(0),
    vertices_(0), tvertices_(0), faces_(0), tfaces_(0), 
    face_normals_(0), vertex_normals_(0) {
    AddChildMoniker("*MESH_NUMVERTEX");
    AddChildMoniker("*MESH_NUMFACES");
    AddChildMoniker("*MESH_VERTEX_LIST");
    AddChildMoniker("*MESH_FACE_LIST");
    AddChildMoniker("*MESH_NUMFACES");
    AddChildMoniker("*MESH_NUMTVERTEX");
    AddChildMoniker("*MESH_TVERTLIST");
    AddChildMoniker("*MESH_NUMTVFACES");
    AddChildMoniker("*MESH_TFACELIST");
    AddChildMoniker("*MESH_NORMALS");
  }
  virtual ~MeshToken() {}

  unsigned GetNumVertices() { return numvertices_; }
  void SetNumVertices(unsigned n) {
    if (n==0)
      ((GeomObjectToken*)parent_)->SetEmpty();
    numvertices_ = n; 
  }

  unsigned GetNumTVertices() { return numtvertices_; }
  void SetNumTVertices(unsigned n) { numtvertices_ = n; }
  
  unsigned GetNumFaces() { return numfaces_; }
  void SetNumFaces(unsigned n) { numfaces_ = n; }

  unsigned GetNumTFaces() { return numtfaces_; }
  void SetNumTFaces(unsigned n) { numtfaces_ = n; }

  vector<double> *GetVertices() { return vertices_; }
  void SetVertices(vector<double> *v) { vertices_ = v; }
  
  vector<unsigned> *GetFaces() { return faces_; }
  void SetFaces(vector<unsigned> *v) { faces_ = v; }
  
  vector<double> *GetTVertices() { return tvertices_; }
  void SetTVertices(vector<double> *v) { tvertices_ = v; }
  
  vector<unsigned> *GetTFaces() { return tfaces_; }
  void SetTFaces(vector<unsigned> *v) { tfaces_ = v; }
  
  vector<double> *GetFaceNormals() { return face_normals_; }
  void SetFaceNormals(vector<double> *v) { face_normals_ = v; }
  
  vector<double> *GetVertexNormals() { return vertex_normals_; }
  void SetVertexNormals(vector<double> *v) { vertex_normals_ = v; }
  
  virtual Token *MakeToken() { return new MeshToken(); }
};



class MeshNumVertexToken : public Token
{

 public:

  MeshNumVertexToken() : Token("*MESH_NUMVERTEX") { nargs_ = 1; }
  virtual ~MeshNumVertexToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned num;
    str >> num;
    ((MeshToken*)parent_)->SetNumVertices(num);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MESH_NUMVERTEX " << ((MeshToken*)parent_)->GetNumVertices()
	<< endl;
  }

  virtual Token *MakeToken() { return new MeshNumVertexToken(); }
};



class MeshNumTVertexToken : public Token
{

 public:

  MeshNumTVertexToken() : Token("*MESH_NUMTVERTEX") { nargs_ = 1; }
  virtual ~MeshNumTVertexToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned num;
    str >> num;
    ((MeshToken*)parent_)->SetNumTVertices(num);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MESH_NUMTVERTEX " << ((MeshToken*)parent_)->GetNumTVertices()
	<< endl;
  }

  virtual Token *MakeToken() { return new MeshNumTVertexToken(); }
};



class MeshNumFacesToken : public Token
{

 public:

  MeshNumFacesToken() : Token("*MESH_NUMFACES") { nargs_ = 1; }
  virtual ~MeshNumFacesToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned num;
    str >> num;
    ((MeshToken*)parent_)->SetNumFaces(num);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MESH_NUMFACES " << ((MeshToken*)parent_)->GetNumFaces()
	<< endl;
  }

  virtual Token *MakeToken() { return new MeshNumFacesToken(); }
};



class MeshNumTVFacesToken : public Token
{

 public:

  MeshNumTVFacesToken() : Token("*MESH_NUMTVFACES") { nargs_ = 1; }
  virtual ~MeshNumTVFacesToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned num;
    str >> num;
    ((MeshToken*)parent_)->SetNumTFaces(num);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MESH_NUMTVFACES " << ((MeshToken*)parent_)->GetNumTFaces()
	<< endl;
  }

  virtual Token *MakeToken() { return new MeshNumTVFacesToken(); }
};



class MeshVertexListToken : public Token
{

 protected:

  vector<double> vertices_;

 public:

  MeshVertexListToken() : Token("*MESH_VERTEX_LIST") {}
  virtual ~MeshVertexListToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned index;
    unsigned numvertices = ((MeshToken*)parent_)->GetNumVertices();
    vertices_.resize(numvertices*3);
    string curstring;
    str >> curstring; // opening delimiter
    DELIMITER delimiter = (DELIMITER)curstring[0];
    str >> curstring;
    while(1) {
      if (curstring == "*MESH_VERTEX") {
	str >> index; // vertex index
	str >> vertices_[index*3]; // X
	str >> vertices_[index*3+1]; // Y
	str >> vertices_[index*3+2]; // Z
	str >> curstring; // get next token (maybe closing delimiter)
      } else {
        if (match(delimiter,(DELIMITER)curstring[0]))
          break;
        else
          str >> curstring;
      }
    }

    ((MeshToken*)parent_)->SetVertices(&vertices_);
    
    return true;
  }

  virtual void Write(ofstream &str) {
    unsigned loop, length;
    
    Indent(str);
    str << "*MESH_VERTEX_LIST {" << endl;
    ++indent_;

    length = vertices_.size();
    for (loop=0; loop<length; loop+=3) {
      Indent(str);
      str << "*MESH_VERTEX " << loop/3 << " "
	  << vertices_[loop] << " " 
	  << vertices_[loop+1] << " " 
	  << vertices_[loop+2] << endl;
    }

    --indent_;
    Indent(str);
    str << "}" << endl;
  }

  vector<double> *GetVertices() { return &vertices_; }

  virtual Token *MakeToken() { return new MeshVertexListToken(); }
};



class MeshTVertListToken : public Token
{

  vector<double> tvertices_;

 public:

  MeshTVertListToken() : Token("*MESH_TVERTLIST") {}
  virtual ~MeshTVertListToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned numtvertices = ((MeshToken*)parent_)->GetNumTVertices();
    tvertices_.resize(numtvertices*3);
    unsigned index;
    string curstring;
    str >> curstring; // opening delimiter
    DELIMITER delimiter = (DELIMITER)curstring[0];
    str >> curstring;
    while(1) {
      if (curstring == "*MESH_TVERT") {
	str >> index; // vertex index
	str >> tvertices_[index*3]; // X
	str >> tvertices_[index*3+1]; // Y
	str >> tvertices_[index*3+2]; // Z
	str >> curstring; // get next token (maybe closing delimiter)
      } else {
        if (match(delimiter,(DELIMITER)curstring[0]))
          break;
        else
          str >> curstring;
      }
    }

    ((MeshToken*)parent_)->SetTVertices(&tvertices_);    
    
    return true;
  }

  virtual void Write(ofstream &str) {
    unsigned loop, length;
    
    Indent(str);
    str << "*MESH_TVERTLIST {" << endl;
    ++indent_;

    length = tvertices_.size();
    for (loop=0; loop<length; loop+=3) {
      Indent(str);
      str << "*MESH_TVERT " << loop/3 << " "
	  << tvertices_[loop] << " " 
	  << tvertices_[loop+1] << " " 
	  << tvertices_[loop+2] << endl;
    }

    --indent_;
    Indent(str);
    str << "}" << endl;
  }

  vector<double> *GetTVertices() { return &tvertices_; }

  virtual Token *MakeToken() { return new MeshTVertListToken(); }
};



class MeshFaceListToken : public Token
{

 protected:

  vector<unsigned> faces_;

 public:

  MeshFaceListToken() : Token("*MESH_FACE_LIST") {}
  virtual ~MeshFaceListToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned numfaces = ((MeshToken*)parent_)->GetNumFaces();
    faces_.resize(numfaces*3);
    string curstring;
    unsigned index;
    str >> curstring; // opening delimiter
    DELIMITER delimiter = (DELIMITER)curstring[0];
    str >> curstring;
    while(1) {
      if (curstring == "*MESH_FACE") {
	str >> index; // face index
	str >> curstring; // :
	str >> curstring; // A:
	str >> faces_[index*3]; // vertex index A
	str >> curstring; // B:
	str >> faces_[index*3+1]; // vertex index B
	str >> curstring; // C:
	str >> faces_[index*3+2]; // vertex index C
	str >> curstring; // AB:
	str >> curstring; // 
	str >> curstring; // BC:
	str >> curstring; // 
	str >> curstring; // CA:
	str >> curstring; //
	str >> curstring; // *MESH_SMOOTHING
	str >> curstring;
	if (curstring != "*MESH_MTLID") 
	  str >> curstring; // *MESH_MTLID
	str >> curstring; // material ID
	str >> curstring; // get next token (maybe closing delimiter)
      } else {
        if (match(delimiter,(DELIMITER)curstring[0]))
          break;
        else
          str >> curstring;
      }
    }
    
#if DEBUG
    cerr << "Token: finished parsing mesh faces" << endl;
#endif
    
    ((MeshToken*)parent_)->SetFaces(&faces_);
    
    return true;
  }

  virtual void Write(ofstream &str) {
    unsigned loop, length;
    
    Indent(str);
    str << "*MESH_FACE_LIST {" << endl;
    ++indent_;

    length = faces_.size();
    for (loop=0; loop<length; loop+=3) {
      Indent(str);
      str << "*MESH_FACE " << loop/3 << ": A: "
	  << faces_[loop] << " B: " 
	  << faces_[loop+1] << " C: " 
	  << faces_[loop+2] 
	  << " AB: 1 BC: 1 CA: 1 *MESH_SMOOTHING 0 *MESH_MTLID 0" << endl;
    }

    --indent_;
    Indent(str);
    str << "}" << endl;
  }

  vector<unsigned> *GetFaces() { return &faces_; }

  virtual Token *MakeToken() { return new MeshFaceListToken(); }
};



class MeshTFaceListToken : public Token
{

  vector<unsigned> tfaces_;

 public:

  MeshTFaceListToken() : Token("*MESH_TFACELIST") {}
  virtual ~MeshTFaceListToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned numtfaces = ((MeshToken*)parent_)->GetNumTFaces();
    tfaces_.resize(numtfaces*3);
    string curstring;
    unsigned index;
    str >> curstring; // opening delimiter
    DELIMITER delimiter = (DELIMITER)curstring[0];
    str >> curstring;
    while(1) {
      if (curstring == "*MESH_TFACE") {
	str >> index; // face index
	str >> tfaces_[index*3]; // A
	str >> tfaces_[index*3+1]; // B
	str >> tfaces_[index*3+2]; // C
	str >> curstring; // get next token (maybe closing delimiter)
      } else {
        if (match(delimiter,(DELIMITER)curstring[0]))
          break;
        else 
          str >> curstring;
      }
    }

#if DEBUG
    cerr << "Token: finished parsing mesh tfaces." << endl;
#endif
    
    ((MeshToken*)parent_)->SetTFaces(&tfaces_);
    
    return true;
  }

  virtual void Write(ofstream &str) {
    unsigned loop, length;
    
    Indent(str);
    str << "*MESH_TFACELIST {" << endl;
    ++indent_;

    length = tfaces_.size();
    for (loop=0; loop<length; loop+=3) {
      Indent(str);
      str << "*MESH_TFACE " << loop/3 << " "
	  << tfaces_[loop] << " " 
	  << tfaces_[loop+1] << " " 
	  << tfaces_[loop+2] << endl;
    }

    --indent_;
    Indent(str);
    str << "}" << endl;
  }

  vector<unsigned> *GetTFaces() { return &tfaces_; }

  virtual Token *MakeToken() { return new MeshTFaceListToken(); }
};



class MeshNormalsToken : public Token
{

  vector<double> face_normals_;
  vector<double> vertex_normals_;

 public:

  MeshNormalsToken() : Token("*MESH_NORMALS") {
    AddChildMoniker("*MESH_FACENORMAL");
    AddChildMoniker("*MESH_VERTEXNORMAL");
  }
  virtual ~MeshNormalsToken() {}

  virtual bool Parse(ifstream &str) {
    unsigned faceindex;
    unsigned numfaces = ((MeshToken*)parent_)->GetNumFaces();
    face_normals_.resize(numfaces*3);
    vertex_normals_.resize(numfaces*9);
    string curstring;
    str >> curstring; // opening delimiter
    DELIMITER delimiter = (DELIMITER)curstring[0];
    str >> curstring;
    while(1) {
      if (curstring == "*MESH_FACENORMAL") {
	str >> faceindex; // face index
	str >> face_normals_[faceindex*3]; // X
	str >> face_normals_[faceindex*3+1]; // Y
	str >> face_normals_[faceindex*3+2]; // Z
	str >> curstring; // *MESH_VERTEXNORMAL
	str >> curstring; // vert index **ignored!**
	str >> vertex_normals_[faceindex*9]; // X1
	str >> vertex_normals_[faceindex*9+1]; // Y1
	str >> vertex_normals_[faceindex*9+2]; // Z1
	str >> curstring; // *MESH_VERTEXNORMAL
	str >> curstring; // vert index **ignored!**
	str >> vertex_normals_[faceindex*9+3]; // X2
	str >> vertex_normals_[faceindex*9+4]; // Y2
	str >> vertex_normals_[faceindex*9+5]; // Z2
	str >> curstring; // *MESH_VERTEXNORMAL
	str >> curstring; // vert index **ignored!**
	str >> vertex_normals_[faceindex*9+6]; // X3
	str >> vertex_normals_[faceindex*9+7]; // Y3
	str >> vertex_normals_[faceindex*9+8]; // Z3
        str >> curstring; // next token (maybe closing delimiter)
      } else {
        if (match(delimiter,(DELIMITER)curstring[0]))
          break;
        else
          str >> curstring;
      }
    }
    
#if DEBUG
    cerr << "Token: finished parsing mesh normals." << endl;
#endif
    
    ((MeshToken*)parent_)->SetFaceNormals(&face_normals_);
    ((MeshToken*)parent_)->SetVertexNormals(&vertex_normals_);
        
    return true;
  }

  virtual void Write(ofstream &str) {
    unsigned loop, length;
    
    Indent(str);
    str << "*MESH_NORMALS {" << endl;
    ++indent_;

    length = ((MeshToken*)parent_)->GetNumFaces();
    for (loop=0; loop<length; ++loop) {
      Indent(str);
      str << "*MESH_FACENORMAL " << loop << " "
	  << face_normals_[loop*3] << " " 
	  << face_normals_[loop*3+1] << " " 
	  << face_normals_[loop*3+2] << endl;
      ++indent_;
      Indent(str);
      str << "*MESH_VERTEXNORMAL 0 "
	  << vertex_normals_[loop*9] << " "
	  << vertex_normals_[loop*9+1] << " "
	  << vertex_normals_[loop*9+2] << endl;
      Indent(str);
      str << "*MESH_VERTEXNORMAL 1 "
	  << vertex_normals_[loop*9+3] << " "
	  << vertex_normals_[loop*9+4] << " "
	  << vertex_normals_[loop*9+5] << endl;
      Indent(str);
      str << "*MESH_VERTEXNORMAL 2 "
	  << vertex_normals_[loop*9+6] << " "
	  << vertex_normals_[loop*9+7] << " "
	  << vertex_normals_[loop*9+8] << endl;
      --indent_;
    }

    --indent_;
    Indent(str);
    str << "}" << endl;
  }

  vector<double> *GetFaceNormals() { return &face_normals_; }
  vector<double> *GetVertexNormals() { return &vertex_normals_; }

  virtual Token *MakeToken() { return new MeshNormalsToken(); }
};



class MaterialRefToken : public Token
{
  
 protected:

  unsigned index_;

 public:

  MaterialRefToken() : Token("*MATERIAL_REF") {}
  virtual ~MaterialRefToken() {}

  virtual bool Parse(ifstream &str) {
    str >> index_;

    ((GeomObjectToken*)parent_)->SetMaterialIndex(index_);
    return true;
  }

  virtual void Write(ofstream &str) {
    Indent(str);
    str << "*MATERIAL_REF " << index_ << endl;
  }

  virtual Token *MakeToken() { return new MaterialRefToken(); }
};



class GroupToken : public Token
{

 public:

  GroupToken() : Token("*GROUP") {
    nargs_ = 1;
    AddChildMoniker("*GEOMOBJECT");
  }
  virtual ~GroupToken() {}

  virtual bool Parse(ifstream &str) {
    ParseArgs(str);
    cout << endl << "Token: start *GROUP " << args_[0] 
	 << " -----------------------" << endl;
    ParseChildren(str);
    cout << "Token: end *GROUP " << args_[0] 
	 << " -----------------------" << endl;
    return true;
  }

  virtual Token *MakeToken() { return new GroupToken(); }
};



class ASEFile : public Token
{

 protected:

  // initialize all possible tokens
  SceneToken A;
  SceneEnvMapToken B;
  SceneAmbientStaticToken C;
  MaterialListToken D;
  MaterialToken E;
  SubMaterialToken F;
  BitmapToken L;
  GeomObjectToken M;
  NodeNameToken N;
  MeshToken O;
  MeshNumVertexToken P;
  MeshNumTVertexToken Q;
  MeshNumFacesToken R;
  MeshNumTVFacesToken S;
  MeshVertexListToken T;
  MeshTVertListToken U;
  MeshFaceListToken X;
  MeshTFaceListToken Y;
  MeshNormalsToken BB;
  MaterialRefToken EE;
  GroupToken FF;
  MaterialNameToken GG;
  MaterialAmbientToken HH;
  MaterialDiffuseToken II;
  MaterialSpecularToken JJ;
  MaterialShineToken KK;
  MaterialTransparencyToken LL;
  MapDiffuseToken MM;
  MapBumpToken NN;
  MapOpacityToken OO;
  MapSelfIllumToken PP;

  bool is_open_;
  ifstream str_;

 public:

  ASEFile(const string &filename) : Token(filename), str_(filename.c_str()) {

    // identify this token as a file
    file_ = true;

    // identify the top level tokens
    AddChildMoniker("*SCENE");
    AddChildMoniker("*MATERIAL_LIST");
    AddChildMoniker("*GEOMOBJECT");
    AddChildMoniker("*GROUP");

    // try to open the given file
    is_open_ = str_.is_open();
  }
  virtual ~ASEFile() {}

  void open(const string& s) {
    moniker_ = s;
    str_.open(moniker_.c_str());
  }

  void close() {
    str_.close();
  }

#if 1
  virtual bool Parse() { 
    if (is_open_) 
      return ((Token*)this)->Parse(str_);
    else
      return false;
  }
#else
  virtual bool Parse() {
    GeomObjectToken A;
    string curstring;
    children_.push_back(&A);
    str_ >> curstring;
    while (curstring != "*GEOMOBJECT")
      str_ >> curstring;
    return A.Parse(str_);
  } 
#endif


  void Write() {
#ifndef _WIN32
    unsigned loop, length;
    int check;
    string dirname, filename;
    char geomcountbuf[100] = "\0";
    int geomcount = 0;

    // create a new directory to hold the new files
    dirname = moniker_ + "-dir";
    check = mkdir(dirname.c_str(),0xffff);
    if (check) {
      cerr << "I/O Error: unable to create directory: " << dirname
	   << ":" << endl << "\t" << strerror(errno) << endl;
      return;
    }

    // create a materials-only file
    filename = dirname + "/materials.ase";
    ofstream matfile(filename.c_str());
    if (!matfile.is_open()) {
      cerr << "I/O Error: unable to create file: " 
	   << filename << endl;
      return;
    }

    // create a scene-only file
    filename = dirname + "/scene.ase";
    ofstream scenefile(filename.c_str());
    if (!scenefile.is_open()) {
      cerr << "I/O Error: unable to create file: " 
	   << filename << endl;
      return;
    }

    // write the children
    length = children_.size();
    for (loop=0; loop<length; ++loop) {

      // write a material list
      if (children_[loop]->GetMoniker() == "*MATERIAL_LIST") {
	indent_ = 0;
	children_[loop]->Write(matfile);
	matfile.close();
      }

      // write a scene
      else if (children_[loop]->GetMoniker() == "*SCENE") {
	indent_ = 0;
	children_[loop]->Write(scenefile);
	scenefile.close();
      }

      // write a GeomObject
      else if (children_[loop]->GetMoniker() == "*GEOMOBJECT") {
	if (((GeomObjectToken*)children_[loop])->Empty())
	  continue;
	geomcount++;
	sprintf(geomcountbuf,"%03d",geomcount);
	string nodename = ((GeomObjectToken*)children_[loop])->GetNodeName();
	filename = dirname + "/" + geomcountbuf + "-" + nodename + ".ase";
	ofstream geomfile(filename.c_str());
	if (!geomfile.is_open()) {
	  cerr << "I/O Error: unable to create file: "
	       << filename << endl;
	  geomcount--;
	  continue;
	}
	
	indent_ = 0;
	geomfile.precision(10);
	children_[loop]->Write(geomfile);
	geomfile.close();
      }

      // write a Group
      else if (children_[loop]->GetMoniker() == "*GROUP") {
	unsigned loop2, length2;
	token_list *children;
	children = children_[loop]->GetChildren();
	length2 = children->size();
	for (loop2=0; loop2<length2; ++loop2) {
	  if (((GeomObjectToken*)(*children)[loop2])->Empty())
	    continue;
	  geomcount++;
	  sprintf(geomcountbuf,"%03d",geomcount);
	  string nodename = 
	    ((GeomObjectToken*)(*children)[loop2])->GetNodeName();
	  filename = dirname + "/" + geomcountbuf + "-" + nodename + ".ase";
	  ofstream geomfile(filename.c_str());
	  if (!geomfile.is_open()) {
	    cerr << "I/O Error: unable to create file: "
		 << filename << endl;
	    geomcount--;
	    continue;
	  }
	  
	  indent_ = 0;
	  geomfile.precision(10);
	  ((*children)[loop2])->Write(geomfile);
	  geomfile.close();
	}
      }
    }
#endif
  }

  virtual Token *MakeToken() { return new ASEFile("no-filename-given"); }

  bool is_open() { return is_open_; }
};



class ASEDir 
{
  
  string dir_;
  vector<ASEFile*> asefiles;

 public:

  ASEDir(const string& s) : dir_(s) {}
  virtual ~ASEDir() {}

  bool Parse() {
    char buf[100];
    string curstring;
    curstring = "ls -1 ";
    curstring += dir_ + "/*.{ase,ASE} > " + dir_ + "/__++temp++__";
    system(curstring.c_str());

    ifstream dirlist((dir_ + "/__++temp++__").c_str());

    if (!dirlist.is_open()) {
      cerr << "I/O Error: unable to open temp file." << endl;
      exit(0);
    }

    dirlist.getline(buf,100);
    while (!dirlist.eof()) {
      asefiles.push_back(new ASEFile(buf));
      dirlist.getline(buf,100);
    }
    dirlist.close();

    curstring = "rm -f ";
    curstring += dir_ + "/__++temp++__";
    system(curstring.c_str());
    
    unsigned length = asefiles.size();
    cout << "found " << length << " ASE files" << endl;
    unsigned loop;
    for (loop=0; loop<length; ++loop) {
      asefiles[loop]->Parse();
    }

    return true;
  }
};

#endif


