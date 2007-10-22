// KInpReader.h
//
// Copyright 2002  Reaction Engineering International
// by Yang
/////////////////////////////////////////////////////

#ifndef KINPREADER_H
#define KINPREADER_H

#pragma warning(disable : 4786)

#include <vector>
#include <string>
#include <map>
#include <set>

typedef struct 
{
	unsigned long secid;
	unsigned long elform;
	double shrf;
	double qr_irid;
	double cst;
	double scoor;
	double nsm;
	double f[8];
} SecBeam;

typedef struct 
{
	unsigned long secid;
	unsigned long elform;
	double shrf;
	double nip;
	double propt;
	double or_irid;
	unsigned long icomp;
	unsigned long setyp;

	double t[4];
	double nloc;
	double marea;

} SecShell;

typedef struct 
{
	unsigned long secid;
	unsigned long elform;
	unsigned long aet;
} SecSolid;

typedef struct
{
	unsigned long eid;
	unsigned long pid;
	unsigned long n[8];
} ElemSolid;

typedef struct
{
	unsigned long eid;
	unsigned long pid;
	unsigned long n[4];
	double t[4];
	double psi;
   double sign;
} ElemShell;

typedef struct
{
	unsigned long eid;
	unsigned long pid;
	unsigned long n1;
	unsigned long n2;
	unsigned long n3;
	unsigned long rt1;
	unsigned long rr1;
	unsigned long rt2;
	unsigned long rr2;
	unsigned long local;
} ElemBeam;

class Node
{
public:
	unsigned long nid;
	double x[3];
	double tc;
	double rc;

   unsigned long nidloc; // part node number used for tri files in solid elements

};

typedef struct
{
	unsigned long n[4];
   std::vector<unsigned long> en;
} face;

typedef struct
{
	unsigned long n[2];
   std::vector<unsigned long> en; 
} edge;

class Part
{
public:
   ~Part();
  std::string title; //card 1
  
  unsigned long pid; //card 2
  unsigned long secid;
  unsigned long mid;
  unsigned long eosid;
  unsigned long hgid;
  unsigned long grav;
  unsigned long adpopt;
  unsigned long tmid;

  unsigned long numsolid;
  unsigned long numshell;

  std::map<std::string, face> faces;
  std::map<unsigned long, ElemShell*> elemShells_m;
  std::vector<face> faces_v;
  std::vector<Node> nodes_v;
  std::map<unsigned int,Node> nodes_m;

};

class k_inp_reader
{

private:
	
  bool match(const std::string& s1, const std::string& s2);	
  std::string getTok(const char* buffer, int start_pos, int len);
  bool isComments(const char* current_line);

 public:
  k_inp_reader();
  //check if this is a key word for the input file	
  int is_inp_keyw(std::string oword);
  //parse the input file to set up the gas system
  int parse_inp(const char* filename);
  void create_tri();
  
  void dump();
  void cross_prod(const double* v1, const double* v2, double* v3);
  void read_vtcs(std::string ptsfile, int& cnt);
  void read_elems(std::string trifile, int cnt);
  void write_uns();
  void swap_4(char* data);
  
  
  ~k_inp_reader();
 
 public: //Data

  ///////////////////////////////////////////////
    std::vector<SecBeam*> secBeams;
    std::vector<SecShell*> secShells;
    std::vector<SecSolid*> secSolids;
    
    std::vector<ElemSolid*> elemSolids;
    std::vector<ElemShell*> elemShells;
    std::vector<ElemBeam*> elemBeams;
    std::vector<Node*> nodes;
    std::vector<Part*> parts;
    
    std::map<unsigned int,Part*> parts_m;
    std::map<unsigned int,Node*> nodes_m;
    std::map<unsigned int,SecShell*> secShells_m;
    
    std::vector<std::vector<int> > elements;
    std::vector<std::vector<float> > vertices;

	double TCORRECTION;
};

#endif
