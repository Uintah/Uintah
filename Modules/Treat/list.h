/* This is a templated linked list implementation designed to make life
 * easier with array-style subscripting access, etc.
 *
 * By: David Fay
 * Last Modified: 9/13/96
 */

#ifndef _LIST_
#define _LIST_

// Some defines that make life easier when Boolean values are ambiguous.

#ifndef BOOL
#define BOOL int
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef NULL
#define NULL 0
#endif

// This is a linked list class designed to have partial functionality as
// as an array.  Keep in mind that it is a linked list, however, and that
// element access is in linear ( O(N) ) time.  Addition and subtraction of
// list elements is also linear.

template <class T>
class List {
public:
  
  List();                                     // The constructor and
  ~List();                                    // destructor
  List(const List<T>& l);                     // The copy constructor
  List(const T* array, const int size);       // A constructor that copies
                                              // a normal array
  List<T>& operator=(const List<T>& l);       // The assignment operator
  BOOL operator==(const List<T>& l) const;    // Equality of Lists
  T & operator[](const int index) const;      // The subscript operator
  BOOL add(const T t, const int index = 0);   // Adds an element to the list
  BOOL add(const List<T>& l, const int index = 0);  // As above, but adds a
                                                    // list to the list
  BOOL remove(const int index, const int num = 1);  // remove element(s)
  BOOL clear();                                     // clean out the list
  const int size() const { return size_of_list; } // the size observer
                                                         // function
private:

  struct element {                     // The basic linked list element.
    T data;
    element* next;
  };

  element* head;                       // pointer to the linked list
  int size_of_list;
  
};

// The constructor.

template <class T>
List<T>::List() {

  size_of_list = 0;
  head = NULL;
}

// The destructor.

template <class T>
List<T>::~List() {

  clear();
}

// A copy constructor.

template <class T>
List<T>::List(const List<T>& l) {

  head = NULL;
  size_of_list = 0;

  add(l);
}

// A copy constructor that copies a normal array into the list.

template <class T>
List<T>::List(const T* array, const int size) {

  element* temp = new element[size];

  for (int i = 0; i < (size - 1); i++) {
    temp[i].data = array[i];
    temp[i].next = &temp[i+1];
  }

  temp[size - 1].data = array[size - 1];
  temp[size - 1].next = NULL;
  head = temp;
  size_of_list = size;
}

// The overloaded assignment operator.  This function must return a reference
// to a list to allow transitive assignments.

template <class T> List<T> &
List<T>::operator=(const List<T>& l) {

  clear();
  add(l);

  return (*this);
}

// The equality operator returns true iff each element in each list satisfies
// the definition of equality for the type T concerned.

template <class T> BOOL
List<T>::operator==(const List<T>& l) const {

  if (size_of_list != l.size_of_list) return FALSE;

  for (int i = 0; i < size_of_list; i++)
    if (!(operator[](i) == l[i])) return FALSE;

  return TRUE;
}

// The overloaded subscript operator allows Lists to be accessed in the
// same way that normal arrays are.  The return type must be a reference
// to ensure that arrays can be used on the left side of assignments.

template <class T> T &
List<T>::operator[](const int index) const {

  element* temp = head;
  
  for (int i = 0; i < index; i++)
    temp = temp->next;

  return temp->data;
}

// The add function inserts an element of type T into the List at the index.
// If no index is specified, the front of the list is used.  TRUE is returned
// if the add is successful, to allow the possible future addition of
// exception handling and error checking.

template <class T> BOOL
List<T>::add(const T t, const int index) {

  element* temp;
  element* e_index = head;
  
  if (index > size_of_list) return FALSE;
  
  if (index == 0) {

    temp = new element;
    temp->data = t;
    temp->next = head;
    head = temp;
    size_of_list++;
    return TRUE;
  }

  for (int i = 0; i < (index - 1); i++)
    e_index = e_index->next;
  
  temp = new element;
  temp->data = t;
  temp->next = e_index->next;
  e_index->next = temp;
  size_of_list++;

  return TRUE;
}

// This function adds a List to the List in question by inserting it at index.
// This function returns TRUE if the operation is successful.

template <class T> BOOL
List<T>::add(const List<T>& l, const int index) {

  element* temp;
  element* e_index = head;

  if (index > size_of_list) return FALSE;

  if (index == 0) {

    temp = new element[l.size_of_list];
    for (int i = 0; i < (l.size_of_list - 1); i++) {
      temp[i].data = l[i];
      temp[i].next = &temp[i+1];
    }

    temp[l.size_of_list - 1].data = l[l.size_of_list - 1];
    temp[l.size_of_list - 1].next = head;
    head = temp;
    size_of_list += l.size_of_list;
    return TRUE;
  }

  for (int i = 0; i < (index - 1); i++)
    e_index = e_index->next;

  temp = new element[l.size_of_list];
  for (int i = 0; i < (l.size_of_list - 1); i++) {
    temp[i].data = l[i];
    temp[i].next = &temp[i+1];
  }
  
  temp[l.size_of_list - 1].data = l[l.size_of_list - 1];
  temp[l.size_of_list - 1].next = e_index->next;
  e_index->next = temp;
  size_of_list += l.size_of_list;

  return TRUE;
}

// This function removes num elements at index.  TRUE is returned upon success.

template <class T> BOOL
List<T>::remove(const int index, const int num) {

  element* temp;
  element* e_index = head;

  if (index > (size_of_list - 1)) return FALSE;
  
  for (int i = 0; i < (index - 1); i++)
    e_index = e_index->next;

  temp = e_index->next;
  e_index->next = e_index->next->next;
  delete temp;
  size_of_list--;

  if (num > 1) return remove(index, num - 1);

  return TRUE;
}

// This function removes all of the elements in the List.  Destruction of
// the data contained in each element (ie the data of type T) is incumbent
// upon the creator of the data.

template <class T> BOOL
List<T>::clear() {

  element* temp;

  while (head != NULL) {
    
    temp = head;
    head = head->next;
    delete temp;
  }

  size_of_list = 0;
  return TRUE;
}

  
#endif
