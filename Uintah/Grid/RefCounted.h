
#ifndef UINTAH_HOMEBREW_REFCOUNTED_H
#define UINTAH_HOMEBREW_REFCOUNTED_H

class RefCounted {
    int refCount;
    int lockIndex;
public:
    RefCounted();
    RefCounted(const RefCounted&);
    RefCounted& operator=(const RefCounted&);
    virtual ~RefCounted();

    void addReference();
    bool removeReference();
};

#endif

