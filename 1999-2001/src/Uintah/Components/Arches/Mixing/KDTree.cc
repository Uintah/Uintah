//----- KDTree.cc --------------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/KDTree.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/InvalidValue.h>

using namespace Uintah::ArchesSpace;

  // Constructor of KD_Tree;
KD_Tree::KD_Tree(int dim, int phiDim){
  d_root = 0;
  d_dim = dim;
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
KD_Tree::Lookup(int key[], vector<REAL>& Phi){
  KD_Node* x = TreeSearch(d_root,key);
  if ( x == 0) {
    //    cout << "Key not found " <<endl;
    return(false);
  }
  Phi = x->Phi;
  return(true);
}


bool
KD_Tree::Insert(int key[], vector<REAL> Phi) {
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
KD_Tree::TreeInsert(KD_Node*& x, int key[], vector<REAL> phi,
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
    DeleteItem(x->right,key,(lev+1)%d_dim);
  }
  else {
    DeleteItem(x->left, key,(lev+1)%d_dim);
  }
}

void                                    
KD_Tree::TreeDelete(KD_Node*& x, int key[]){
  vector<REAL> phi;
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
KD_Tree::ProcessLeftMost(KD_Node*& x, vector<REAL>& phi){
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
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//

