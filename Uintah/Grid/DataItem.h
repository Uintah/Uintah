
#ifndef UINTAH_HOMEBREW_DataItem_H
#define UINTAH_HOMEBREW_DataItem_H

class Region;

class DataItem {
public:

    virtual ~DataItem();
    virtual void get(DataItem&) const = 0;
    virtual DataItem* clone() const = 0;
    virtual void allocate(const Region*) = 0;
protected:
    DataItem(const DataItem&);
    DataItem();

private:
    DataItem& operator=(const DataItem&);
};

#endif
