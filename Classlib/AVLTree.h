
/*
 *  AVLTree.h: Interface to AVLTree class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Classlib_AVLTree_h
#define SCI_Classlib_AVLTree_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <iostream.h>

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

#endif /* SCI_Classlib_AVLTree_h */
