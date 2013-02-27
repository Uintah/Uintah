/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


/*
 *  AVLTree.h: Interface to AVLTree class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 */

#ifndef SCI_Containers_AVLTree_h
#define SCI_Containers_AVLTree_h 1

#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

template<class Key, class Data> class AVLTree;
template<class Key, class Data> class AVLTreeIter;

template <class Key, class Data>
class TreeLink  {
    friend class AVLTree<Key, Data>;
    friend class AVLTreeIter<Key, Data>;
    Key key;
    Data data;
    TreeLink *left;
    TreeLink *right;
    TreeLink *parent;
    int balance_factor;
    int deleted;
    TreeLink(const Key& key, const Data& data);
    ~TreeLink();
};

template<class Key, class Data>
class AVLTree {
    TreeLink<Key, Data> *root;
    int nitems;
    void remove_node(TreeLink<Key, Data>*);
    void fillin_array(TreeLink<Key, Data>*, TreeLink<Key, Data>**, int&);
public:
    friend class AVLTreeIter<Key, Data>;
    AVLTree();
    AVLTree(const AVLTree<Key, Data>&);
    AVLTree<Key, Data>& operator=(const AVLTree<Key, Data>&);
    ~AVLTree();
    void insert(const Key& key, const Data& data);
    int lookup(const Key& key, Data& data);
    void remove(const Key& key);
    void remove(const Key& key, const Data& data);
    void remove(const AVLTreeIter<Key, Data>&);
    void remove_all();
    void cleanup();
    int size();
    Data pop();
};

template<class Key, class Data>
class AVLTreeIter {
    AVLTree<Key, Data> *tree;
    TreeLink<Key, Data> *current;
    friend class AVLTree<Key, Data>;
public:
    AVLTreeIter(AVLTree<Key, Data> *_tree);
    ~AVLTreeIter();
    int search(const Key& key);
    Key get_key();
    Data get_data();
    void first();
    int ok();
    void operator++();
    void operator--();
};

template<class Key, class Data>
TreeLink<Key, Data>::TreeLink(const Key& key, const Data& data)
: key(key), data(data), left(0), right(0), parent(0), 
  balance_factor(0), deleted(0)
{
}

template<class Key, class Data>
TreeLink<Key, Data>::~TreeLink()
{
    if(left)delete left;
    if(right)delete right;
}

template<class Key, class Data>
AVLTree<Key, Data>::AVLTree()
{
    root=0;
    nitems=0;
}

template<class Key, class Data>
AVLTree<Key, Data>::~AVLTree()
{
    // Delete all nodes
    if(root)delete root;	// Recursively deletes everything
}

template <class Key, class Data>
void AVLTree<Key, Data>::insert(const Key& key, const Data& data)
{
    TreeLink<Key, Data>* newnode=scinew TreeLink<Key, Data>(key, data);
    nitems++;
    newnode->left=newnode->right=newnode->parent=0;
    newnode->deleted=newnode->balance_factor=0;
    if(!root){
	// Start of tree
	root=newnode;
	return;
    }
    TreeLink<Key, Data> *lastunbalancednode=root;
    TreeLink<Key, Data> *parent=0;
    TreeLink<Key, Data> *current=root;
    while(current!=0){
	if(current->deleted && !current->right && !current->left){
	    /* Put it here... */
	    current->key=newnode->key;
	    current->data=newnode->data;
	    current->deleted=0;
	    delete newnode;
	    return;
	}
	if(current->balance_factor != 0){
	    lastunbalancednode=current;
	}
	parent=current;
	if(newnode->key < current->key){
	    current=current->left;
	} else {
	    current=current->right;
	}
    }
    newnode->parent=parent;
    if(newnode->key < parent->key){
	// Insert on left
	parent->left=newnode;
    } else {
	// Insert on right
	parent->right=newnode;
    }
    // Adjust balance factors of nodes on the path between
    // lastunbalancednode and parent
    int bf;
    if(newnode->key < lastunbalancednode->key){
	current=lastunbalancednode->left;
	bf=1;
    } else {
	current=lastunbalancednode->right;
	bf=-1;
    }
    TreeLink<Key, Data> *savenode=current;
    while(current != newnode){
	if(newnode->key < current->key){
	    // height of left increases by 1
	    current->balance_factor=1;
	    current=current->left;
	} else {
	    // height of right increases by 1
	    current->balance_factor=-1;
	    current=current->right;
	}
    }
    // See if the tree is unbalanced
    if(lastunbalancednode->balance_factor == 0){
	// Tree is still balanced
	lastunbalancednode->balance_factor=bf;
    } else if(lastunbalancednode->balance_factor+bf == 0){
	// Still balanced
	lastunbalancednode->balance_factor=0;
    } else {
	// Tree is still unbalanced, determine rotation type
	//TreeLink<Key, Data> *oldchild=lastunbalancednode;
	TreeLink<Key, Data> *newroot=0;
	if(bf == 1){
	    if(savenode->balance_factor==1){
		// Rotation LL
		savenode->parent=lastunbalancednode->parent;
		lastunbalancednode->left=savenode->right;
		if(lastunbalancednode->left)
		    lastunbalancednode->left->parent=lastunbalancednode;
		savenode->right=lastunbalancednode;
		lastunbalancednode->parent=savenode;
		lastunbalancednode->balance_factor=0;
		savenode->balance_factor=0;
		newroot=savenode;
	    } else {
		// Rotation LR
		TreeLink<Key, Data> *savenode2=savenode->right;
		savenode2->parent=lastunbalancednode->parent;
		savenode->right=savenode2->left;
		if(savenode->right)savenode->right->parent=savenode;
		lastunbalancednode->left=savenode2->right;
		if(lastunbalancednode->left)
		    lastunbalancednode->left->parent=lastunbalancednode;
		savenode2->left=savenode;
		savenode->parent=savenode2;
		savenode2->right=lastunbalancednode;
		lastunbalancednode->parent=savenode2;
		if(savenode2->balance_factor == 1){
		    lastunbalancednode->balance_factor=-1;
		    savenode->balance_factor=0;
		} else if(savenode2->balance_factor == -1){
		    lastunbalancednode->balance_factor=0;
		    savenode->balance_factor=1;
		} else {
		    lastunbalancednode->balance_factor=0;
		    savenode->balance_factor=0;
		}
		savenode2->balance_factor=0;
		newroot=savenode2;
	    }
	} else {
	    if(savenode->balance_factor==-1){
		// Rotation RR
		savenode->parent=lastunbalancednode->parent;
		lastunbalancednode->right=savenode->left;
		if(lastunbalancednode->right)
		    lastunbalancednode->right->parent=lastunbalancednode;
		savenode->left=lastunbalancednode;
		lastunbalancednode->parent=savenode;
		lastunbalancednode->balance_factor=0;
		savenode->balance_factor=0;
		newroot=savenode;
	    } else {
		// Rotation RL
		TreeLink<Key, Data> *savenode2=savenode->left;
		savenode2->parent=lastunbalancednode->parent;
		savenode->left=savenode2->right;
		if(savenode->left)savenode->left->parent=savenode;
		lastunbalancednode->right=savenode2->left;
		if(lastunbalancednode->right)
		    lastunbalancednode->right->parent=lastunbalancednode;
		savenode2->right=savenode;
		savenode->parent=savenode2;
		savenode2->left=lastunbalancednode;
		lastunbalancednode->parent=savenode2;
		if(savenode2->balance_factor == -1){
		    lastunbalancednode->balance_factor=1;
		    savenode->balance_factor=0;
		} else if(savenode2->balance_factor == 1){
		    lastunbalancednode->balance_factor=0;
		    savenode->balance_factor=-1;
		} else {
		    lastunbalancednode->balance_factor=0;
		    savenode->balance_factor=0;
		}
		savenode2->balance_factor=0;
		newroot=savenode2;
	    }
	}
	// Now put new tree in place
	if(newroot->parent == 0){
	    root=newroot;
	    newroot->parent=0;
	} else if(newroot->parent->left==lastunbalancednode){
	    // Replace left pointer
	    newroot->parent->left=newroot;
	} else if(newroot->parent->right==lastunbalancednode){
	    // Replace right pointer
	    newroot->parent->right=newroot;
	}
    }
}

template<class Key, class Data>
int AVLTree<Key, Data>::lookup(const Key& key, Data& data)
{
    TreeLink<Key, Data>* current=root;
    while(current != 0){
	if(!current->deleted && key == current->key){
	    data=current->data;
	    return(1);
	} else if(key < current->key){
	    current=current->left;
	} else {
	    current=current->right;
	}
    }
    return 0;
}

template<class Key, class Data>
AVLTreeIter<Key, Data>::AVLTreeIter(AVLTree<Key, Data> *tree)
: tree(tree)
{
    current=0;
}

template<class Key, class Data>
AVLTreeIter<Key, Data>::~AVLTreeIter()
{
}

template<class Key, class Data>
int AVLTreeIter<Key, Data>::search(const Key& key)
{
    current=tree->root;
    TreeLink<Key, Data>* last=current;
    while(current != 0){
	if(!current->deleted && key == current->key){
	    return(1);
	} else if(key < current->key){
	    current=current->left;
	} else {
	    last=current;
	    current=current->right;
	}
    }
    current=last;
    return 0;
}

template<class Key, class Data>
Key AVLTreeIter<Key, Data>::get_key()
{
    return current->key;
}

template<class Key, class Data>
Data AVLTreeIter<Key, Data>::get_data()
{
    return current->data;
}

template<class Key, class Data>
void AVLTreeIter<Key, Data>::first()
{
    current=tree->root;
    if(!current)return;
    while(current->left)current=current->left;
    if(current && current->deleted)operator++();
}

template<class Key, class Data>
int AVLTreeIter<Key, Data>::ok()
{
    if(tree->root==0 || current==0)return 0;
    return 1;
}

template<class Key, class Data>
void AVLTreeIter<Key, Data>::operator++()
{
    ASSERT(current != 0);
    do {
	if(current->right){
	    current=current->right;
	    while(current->left)current=current->left;
	} else {
	// Move up tree to find a right link
	    while(current->parent && current==current->parent->right){
		current=current->parent;
	    }
	    current=current->parent;
	}
    } while(current && current->deleted);
}


template<class Key, class Data>
void AVLTreeIter<Key, Data>::operator--()
{
    ASSERT(current != 0);
    do {
	if(current->left){
	    current=current->left;
	    while(current->right)current=current->right;
	} else {
	    // Move up tree to find a right link
	    while(current->parent && current==current->parent->left){
		current=current->parent;
	    }
	    current=current->parent;
	}
    } while(current && current->deleted);
}


template<class Key, class Data>
int AVLTree<Key, Data>::size()
{
    return nitems;
}

template<class Key, class Data>
Data AVLTree<Key, Data>::pop()
{
    ASSERT(root != 0);
    TreeLink<Key, Data>* rem=root;
    while(rem->left)rem=rem->left;
    while(rem && rem->deleted){
	if(rem->right){
	    rem=rem->right;
	    while(rem->left)rem=rem->left;
	} else {
	    // Move up tree to find a right link
	    while(rem->parent && rem==rem->parent->right){
		rem=rem->parent;
	    }
	    rem=rem->parent;
	}
    }
    ASSERT(rem != 0);
    Data data(rem->data);
    remove_node(rem);
    return data;
}

template<class Key, class Data>
void AVLTree<Key, Data>::remove(const Key& key)
{
    int removed_one=1;
    while(removed_one){
	TreeLink<Key, Data>* current=root;
	removed_one=0;
	while(current != 0 && !removed_one){
	    if(!current->deleted && key == current->key){
		remove_node(current);
		removed_one=1;
	    } else if(key < current->key){
		current=current->left;
	    } else {
		current=current->right;
	    }
	}
    }
}

template<class Key, class Data>
void AVLTree<Key, Data>::remove(const Key& key, const Data& data)
{
    int removed_one=1;
    while(removed_one){
	TreeLink<Key, Data>* current=root;
	removed_one=0;
	while(current != 0 && !removed_one){
	    if(!current->deleted && key == current->key
	       && data==current->data){
		    remove_node(current);
		    removed_one=1;
		    current=0;
	    } else if(key < current->key){
		current=current->left;
	    } else {
		current=current->right;
	    }
	}
    }
}

template<class Key, class Data>
void AVLTree<Key, Data>::remove(const AVLTreeIter<Key, Data>& iter)
{
    ASSERT(iter.current != 0);
    remove_node(iter.current);
}

template<class Key, class Data>
void AVLTree<Key, Data>::remove_node(TreeLink<Key, Data>* node)
{
    nitems--;
    node->deleted=1;
    // We should do this right someday???
}    

// This rebuilds the tree, removing all of the "deleted" nodes
// This should be unnecessary if we ever get around to doing the
// deletions correctly
template<class Key, class Data>
void AVLTree<Key, Data>::cleanup()
{
    if(!root)return;
    TreeLink<Key, Data>** oldnodes=scinew TreeLink<Key, Data>*[nitems];
    int idx=0;
    fillin_array(root, oldnodes, idx);
    int n=nitems;
    nitems=0;
    root=0;
    for(int i=0;i<n;i++)
	insert(oldnodes[i]->key, oldnodes[i]->data);
    delete[] oldnodes;
}

template<class Key, class Data>
void AVLTree<Key, Data>::fillin_array(TreeLink<Key, Data>* p,
				      TreeLink<Key, Data>** array, int& i)
{
    if(!p->deleted)
	array[i++]=p;
    if(p->left)
	fillin_array(p->left, array, i);
    if(p->right)
	fillin_array(p->right, array, i);
    if(p->deleted){
	p->left=p->right=0;
	delete p;
    }
}

template<class Key, class Data>
void AVLTree<Key, Data>::remove_all()
{
    if(root)
	delete root;
    root=0;
    nitems=0;
}

} // End namespace SCIRun


#endif /* SCI_Containers_AVLTree_h */
