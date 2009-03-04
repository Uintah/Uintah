/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef HASHTABLEENTRY_H
#define HASHTABLEENTRY_H 1

namespace rtrt {

template<class Key, class Data> class HashTable;
template<class Key, class Data> class HashTableIter;

template<class Key, class Data> class HashTableEntry {
    /*
     * Copy constructor intentionally left undefined.
     */
    HashTableEntry(const HashTableEntry<Key, Data>&);

    friend class HashTable<Key, Data>;
    friend class HashTableIter<Key, Data>;
    Key key;
    Data data;
    HashTableEntry<Key, Data>* next;
    
    HashTableEntry();
    HashTableEntry(const Key& key, const Data& data,
		   HashTableEntry<Key, Data>* next );
};

  /////////////////////////////////////////
  // All the templated functions

template<class Key, class Data>
HashTableEntry<Key, Data>::HashTableEntry()
    : next(0)
{
}

template<class Key, class Data>
HashTableEntry<Key, Data>::HashTableEntry(const Key& key, const Data& data,
					  HashTableEntry<Key, Data>* next )
    : key(key), data(data), next(next)
{
}
  
} // end namespace rtrt

#endif
