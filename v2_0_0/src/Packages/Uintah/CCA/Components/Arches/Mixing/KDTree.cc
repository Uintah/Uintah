//----- KDTree.cc --------------------------------------------------

/* REFERENCED */

#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <iostream>
using namespace std;
using namespace Uintah;

  // Constructor of KD_Tree;
KD_Tree::KD_Tree(int dim, int phiDim){
  d_root = 0;
  d_dim = dim;
  //cout << "KDTree::d_dim = "<<d_dim<<phiDim<<endl;
  d_dimStateSpaceVars = phiDim;
}

KD_Tree::~KD_Tree(){
  DestroyTree(d_root);
}     


int 
KD_Tree::getSize() const{
  return(d_size);
}

    
   /**
   * Report if the Tree is empty.
   */

bool
KD_Tree::IsEmpty () const{
  return(d_size == 0);
}

bool
//KD_Tree::Lookup(int key[], vector<double>& Phi){
KD_Tree::Lookup(int key[], Stream& Phi){
  //cout<<"KDTree::Lookup"<<endl;
  KD_Node* x = TreeSearch(d_root,key);
  if ( x == 0) {
    //    cout << "Key not found " <<endl;
    return(false);
  }
  //cout <<"KDTree::lookup was successful"<<endl;
  Phi = x->Phi;
  return(true);
}


bool
//KD_Tree::Insert(int key[], vector<double>& Phi) {
KD_Tree::Insert(int key[], Stream& Phi) {
    // returns true if the inserted node has correct Phi
  return(TreeInsert(d_root, key, Phi, 0)->Phi == Phi);
}
  
bool
KD_Tree::Delete(int key[]){
  KD_Node *x = new KD_Node;
  x = TreeSearch(d_root,key);
  if( x == 0) {
    //      cout << " the Key not found" << endl;
    return(false);
  }
  DeleteItem(d_root,key,0) ;     //changed
  return(true);
}


KD_Node*
KD_Tree::TreeSearch(KD_Node *x, int key[]){
  
  int i;
  if ( x == 0) return(x);
  for(int lev=0; x != 0; lev=(lev+1)%d_dim){
    for(i=0; i<d_dim && key[i] == x->keys[i]; i++);
    if(i==d_dim) return(x);
    if(key[lev] > x->keys[lev]) x = x->right;
    else  x = x->left; 
  }
  return(x);
}

KD_Node* 
//KD_Tree::TreeInsert(KD_Node*& x, int key[], vector<double> phi,
KD_Tree::TreeInsert(KD_Node*& x, int key[], Stream phi,
		    int lev){
  int i;
  if(x==0) { 
    x = new KD_Node(d_dim, d_dimStateSpaceVars,key,phi,0,0);
    d_size++;
  }
  else {
    for(i=0; i<d_dim && key[i]==x->keys[i]; i++);
    if(i== d_dim) {
      cout << "the node already exist" << endl;
      for (int ii = 0; ii < d_dim; ii++) {
	cout.width(10);
	cout << key[ii] << " " ; 
	if (!(ii % 10)) cout << endl; 
      }
      x->Phi = phi;
    }
    else if (key[lev] > x->keys[lev]){
      x->right = TreeInsert(x->right, key, phi, (lev+1)%d_dim);}
    else x->left = TreeInsert(x->left, key, phi, (lev+1)%d_dim);
  }
  return(x);
} 

bool
KD_Tree::DeleteItem(KD_Node*& x, int key[], int lev){
  int i ;
  for(i=0; i<d_dim && key[i] == x->keys[i]; i++);
  if(i==d_dim) {
    TreeDelete(x, key);
    d_size--;
    return(true);
  }
  if(key[lev] > x->keys[lev]) {
    return DeleteItem(x->right,key,(lev+1)%d_dim);
  }
  else {
    return DeleteItem(x->left, key,(lev+1)%d_dim);
  }
}

void                                    
KD_Tree::TreeDelete(KD_Node*& x, int* /* key[] */ ){
  //vector<double> phi;
  Stream phi;
  KD_Node *z = new KD_Node;
  // If Node x is a leaf
  if((x->left==0) && (x->right==0)){
    delete x;
    x = 0;
  }
  // If x has no left child
  else if ( x->left==0){
    z = x;
    x = x->right;
    z->right = 0;
    delete z;
  }   
  // If x has no right child
  else if (x->right==0){
    z = x;
    x = z->left;
    z->left = 0;
    delete z;
    z = 0;
  }
  // If x has two children: retrieve and delete the inorder successor
  else {
    ProcessLeftMost(x->right,phi);
    x->Phi = phi;
  }
}
                
void 
//KD_Tree::ProcessLeftMost(KD_Node*& x, vector<double>& phi){
KD_Tree::ProcessLeftMost(KD_Node*& x, Stream& phi){
  KD_Node *z = new KD_Node;
  if(x->left==0){
    phi = x->Phi;
    z = x;
    x = x->right;
    z->right = 0;
    delete z;
  }
  else 
    ProcessLeftMost(x->left,phi);
}     
 
void 
KD_Tree::DestroyTree(KD_Node *x){
  if(x != 0){
    DestroyTree(x->left);
    DestroyTree(x->right);
    delete x;
    x = 0;
  }
}


//
// $Log$
// Revision 1.8  2003/01/31 18:22:46  spinti
// Removed conversion from stream to vector. Tables now stored as vector of streams.
//
// Revision 1.7  2002/05/31 22:04:44  spinti
// *** empty log message ***
//
// Revision 1.6  2001/11/08 19:13:44  spinti
// 1. Corrected minor problems in ILDMReactionModel.cc
// 2. Added tabulation capability to StanjanEquilibriumReactionModel.cc. Now,
//    a reaction table is created dynamically. The limits and spacing in the
//    table are specified in the *.ups file.
// 3. Corrected the mixture temperature computation in Stream::addStream. It
//    now is computed using a Newton search.
// 4. Made other minor corrections to various reaction model files.
//
// Revision 1.3  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// Revision 1.2  2001/02/02 01:54:34  rawat
// cnges made for checkpointing to work
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//

