/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  MaskedLatVolMesh.h:
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   Feb 2003
 *
 *
 */



#ifndef SCI_project_MaskedLatVolMesh_h
#define SCI_project_MaskedLatVolMesh_h 1

#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>
#include <vector>
#include <set>
#include <sci_comp_warn_fixes.h>

namespace SCIRun {

using std::string;

template <class Basis>
class MaskedLatVolMesh : public LatVolMesh<Basis>
{
public:
  typedef Basis                                   basis_type;
  typedef LockingHandle<MaskedLatVolMesh<Basis> > handle_type;

  struct MaskedLatIndex;
  friend struct MaskedLatIndex;

  struct MaskedLatIndex
  {
  public:
    MaskedLatIndex() : i_(0), j_(0), k_(0), mesh_(0) {}
    MaskedLatIndex(const MaskedLatVolMesh *m, unsigned int i, unsigned int j,
                   unsigned int k) : i_(i), j_(j), k_(k), mesh_(m) {}


    std::ostream& str_render(std::ostream& os) const
    {
      os << "[" << i_ << "," << j_ << "," << k_ << "]";
      return os;
    }

    unsigned int i_, j_, k_;

    // Needs to be here so we can compute a sensible index.
    const MaskedLatVolMesh *mesh_;
  };


  struct CellIndex : public MaskedLatIndex
  {
    CellIndex() : MaskedLatIndex() {}
    CellIndex(const MaskedLatVolMesh *m,
              unsigned int i, unsigned int j, unsigned int k)
      : MaskedLatIndex(m,i,j,k) {}

    // The 'operator unsigned()' cast is used to convert a CellIndex
    // into a single scalar, in this case an 'index' value that is used
    // to index into a field.
    operator unsigned() const {
      ASSERT(this->mesh_);
      return this->i_ + (this->mesh_->ni_-1)*this->j_ +
        (this->mesh_->ni_-1)*(this->mesh_->nj_-1)*this->k_;
    }

    bool operator ==(const CellIndex &a) const
    {
      return (this->i_ == a.i_ && this->j_ == a.j_ &&
              this->k_ == a.k_ && this->mesh_ == a.mesh_);
    }

    bool operator !=(const CellIndex &a) const
    {
      return !(*this == a);
    }

    static string type_name(int i=-1)
    { ASSERT(i<1); return "MaskedLatVolMesh::CellIndex"; }
  };


  struct NodeIndex : public MaskedLatIndex
  {
    NodeIndex() : MaskedLatIndex() {}
    NodeIndex(const MaskedLatVolMesh *m, unsigned int i,
              unsigned int j, unsigned int k) :
      MaskedLatIndex(m,i,j,k)
    {}

    // The 'operator unsigned()' cast is used to convert a NodeIndex
    // into a single scalar, in this case an 'index' value that is used
    // to index into a field.
    operator unsigned() const {
      ASSERT(this->mesh_);
      return this->i_ + this->mesh_->ni_*this->j_ +
        this->mesh_->ni_*this->mesh_->nj_*this->k_;
    }


    bool operator ==(const MaskedLatIndex &a) const
    {
      return this->i_ == a.i_ && this->j_ == a.j_ && this->k_ == a.k_ && this->mesh_ == a.mesh_;
    }

    bool operator !=(const MaskedLatIndex &a) const
    {
      return !(*this == a);
    }

    static string type_name(int i=-1)
    { ASSERT(i<1); return "MaskedLatVolMesh::NodeIndex"; }
  };


  struct EdgeIndex : public MaskedLatIndex
  {
    EdgeIndex() : MaskedLatIndex(), dir_(0) {}
    EdgeIndex(const MaskedLatVolMesh *m,
              unsigned int i, unsigned int j, unsigned int k, unsigned int dir)
      : MaskedLatIndex(m, i,j,k) , dir_(dir){}

    // The 'operator unsigned()' cast is used to convert a EdgeIndex
    // into a single scalar, in this case an 'index' value that is used
    // to index into a field.
    operator unsigned() const {
      ASSERT(this->mesh_);
      switch (dir_)
      {
      case 0: return (this->i_ + (this->mesh_->ni_-1)*this->j_ +
                      (this->mesh_->ni_-1)*this->mesh_->nj_*this->k_);
      case 1: return (this->j_ + (this->mesh_->nj_-1)*this->k_ +
                      (this->mesh_->nj_-1)*this->mesh_->nk_*this->i_ +
                      (this->mesh_->ni_-1)*this->mesh_->nj_*this->mesh_->nk_);
      case 2: return (this->k_ + (this->mesh_->nk_-1)*this->i_ +
                      (this->mesh_->nk_-1)*this->mesh_->ni_*this->j_ +
                      (this->mesh_->ni_-1)*this->mesh_->nj_*this->mesh_->nk_ +
                      this->mesh_->ni_*(this->mesh_->nj_-1)*this->mesh_->nk_);
      default: return 0;
      }
    }

    bool operator ==(const EdgeIndex &a) const
    {
      return (this->i_ == a.i_ && this->j_ == a.j_ && this->k_ == a.k_ &&
              this->mesh_ == a.mesh_ && this->dir_ == a.dir_);
    }

    bool operator !=(const EdgeIndex &a) const
    {
      return !(*this == a);
    }

    static string type_name(int i=-1) {
      ASSERT(i<1); return "MaskedLatVolMesh::EdgeIndex";
    }

    unsigned int dir_;
  };


  struct FaceIndex : public MaskedLatIndex
  {
    FaceIndex() : MaskedLatIndex() {}
    FaceIndex(const MaskedLatVolMesh *m,
              unsigned int i, unsigned int j,
              unsigned int k, unsigned int dir) :
      MaskedLatIndex(m, i,j,k),
      dir_(dir)
    {}

    // The 'operator unsigned()' cast is used to convert a FaceIndex
    // into a single scalar, in this case an 'index' value that is used
    // to index into a field.
    operator unsigned() const {
      ASSERT(this->mesh_);
      switch (dir_)
      {
      case 0: return (this->i_ + (this->mesh_->ni_-1)*this->j_ +
                      (this->mesh_->ni_-1)*(this->mesh_->nj_-1)*this->k_);
      case 1: return (this->j_ + (this->mesh_->nj_-1)*this->k_ +
                      (this->mesh_->nj_-1)*(this->mesh_->nk_-1)*this->i_ +
                      (this->mesh_->ni_-1)*(this->mesh_->nj_-1)*
                      this->mesh_->nk_);
      case 2: return (this->k_ + (this->mesh_->nk_-1)*this->i_ +
                      (this->mesh_->nk_-1)*(this->mesh_->ni_-1)*this->j_ +
                      (this->mesh_->ni_-1)*(this->mesh_->nj_-1)*
                      this->mesh_->nk_ + this->mesh_->ni_*
                      (this->mesh_->nj_-1)*(this->mesh_->nk_-1));
      default: return 0; //ASSERTFAIL("MLVMFaceIndex dir_ off.");
      }
    }

    bool operator ==(const FaceIndex &a) const
    {
      return (this->i_ == a.i_ && this->j_ == a.j_ && this->k_ == a.k_ &&
              this->mesh_ == a.mesh_ && this->dir_ == a.dir_);
    }

    bool operator !=(const FaceIndex &a) const
    {
      return !(*this == a);
    }

    static string type_name(int i=-1) {
      ASSERT(i<1); return "MaskedLatVolMesh::FaceIndex";
    }

    unsigned int dir_;
  };


  struct CellSize : public MaskedLatIndex
  {
  public:
    CellSize() : MaskedLatIndex() {}
    CellSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned() const
    {
      return this->i_*this->j_*this->k_;
    }
  };

  struct NodeSize : public MaskedLatIndex
  {
  public:
    NodeSize() : MaskedLatIndex() {}
    NodeSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned() const
    {
      return this->i_*this->j_*this->k_;
    }
  };


  struct EdgeSize : public MaskedLatIndex
  {
  public:
    EdgeSize() : MaskedLatIndex() {}
    EdgeSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned() const
    {
      return (this->i_-1)*this->j_*this->k_ + this->i_*(this->j_-1)*this->k_ + this->i_*this->j_*(this->k_-1);
    }
  };


  struct FaceSize : public MaskedLatIndex
  {
  public:
    FaceSize() : MaskedLatIndex() {}
    FaceSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned() const
    {
      return this->i_*(this->j_-1)*(this->k_-1) + (this->i_-1)*this->j_*(this->k_-1) + (this->i_-1)*(this->j_-1)*this->k_;
    }
  };


  struct NodeIter : public MaskedLatIndex
  {
  public:
    NodeIter() : MaskedLatIndex() {}
    NodeIter(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : MaskedLatIndex(m, i, j, k) {}

    const NodeIndex &operator *() const
    {
      return (const NodeIndex&)(*this);
    }

    operator unsigned()  const
    {
      ASSERT(this->mesh_);
      return this->i_ + this->mesh_->ni_*this->j_ + this->mesh_->ni_*this->mesh_->nj_*this->k_;
    }

    NodeIter &operator++() {
      do next(); while (!this->mesh_->check_valid(*this) &&
                        (this->k_ < (this->mesh_->min_k_+this->mesh_->nk_)));
      return *this;
    }
    NodeIter &operator--() {
      do prev(); while (!this->mesh_->check_valid(*this));
      return *this;
    }

    void next() {
      this->i_++;
      if (this->i_ >= this->mesh_->min_i_+this->mesh_->ni_) {
        this->i_ = this->mesh_->min_i_;
        this->j_++;
        if (this->j_ >=  this->mesh_->min_j_+this->mesh_->nj_) {
          this->j_ = this->mesh_->min_j_;
          this->k_++;
        }
      }
    }
    void prev() {
      if (this->i_ == this->mesh_->min_i_) {
        this->i_ = this->mesh_->min_i_ + this->mesh_->ni_;
        if (this->j_ == this->mesh_->min_j_) {
          this->j_ = this->mesh_->min_j_ + this->mesh_->nj_;
          ASSERTMSG(this->k_ != this->mesh_->min_k_-1, "Cant prev() from first node!");
          this->k_--;
        }
        else {
          this->j_--;
        }
      }
      else {
        this->i_--;
      }
    }


  private:
    NodeIter operator++(int)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }
    NodeIter operator--(int)
    {
      NodeIter result(*this);
      operator--();
      return result;
    }
  };


  struct EdgeIter : public EdgeIndex
  {
  public:
    EdgeIter() : EdgeIndex() {}
    EdgeIter(const MaskedLatVolMesh *m,
             unsigned int i, unsigned int j,
             unsigned int k, unsigned int dir)
      : EdgeIndex(m, i, j, k,dir) {}

    const EdgeIndex &operator *() const
    {
      return (const EdgeIndex&)(*this);
    }

    bool operator ==(const EdgeIter &a) const
    {
      return (this->i_ == a.i_ && this->j_ == a.j_ && this->k_ == a.k_ &&
              this->mesh_ == a.mesh_ && this->dir_ == a.dir_);
    }

    bool operator !=(const EdgeIter &a) const
    {
      return !(*this == a);
    }

    EdgeIter &operator++() {
      do next(); while (!this->mesh_->check_valid(*this) && this->dir_ < 3);
      return *this;
    }
    EdgeIter &operator--() {
      do prev(); while (!this->mesh_->check_valid(*this));
      return *this;
    }

    void next() {
      switch (this->dir_)
      {
      case 0:
        this->i_++;
        if (this->i_ >= this->mesh_->min_i_+this->mesh_->ni_-1) {
          this->i_ = this->mesh_->min_i_;
          this->j_++;
          if (this->j_ >=  this->mesh_->min_j_+this->mesh_->nj_) {
            this->j_ = this->mesh_->min_j_;
            this->k_++;
            if (this->k_ >= this->mesh_->min_k_+this->mesh_->nk_) {
              this->dir_++;
              this->i_ = 0;
              this->j_ = 0;
              this->k_ = 0;
            }
          }
        }
        break;
      case 1:
        this->j_++;
        if (this->j_ >= this->mesh_->min_j_+this->mesh_->nj_-1) {
          this->j_ = this->mesh_->min_j_;
          this->k_++;
          if (this->k_ >=  this->mesh_->min_k_+this->mesh_->nk_) {
            this->k_ = this->mesh_->min_k_;
            this->i_++;
            if (this->i_ >= this->mesh_->min_i_+this->mesh_->ni_) {
              this->dir_++;
              this->i_ = 0;
              this->j_ = 0;
              this->k_ = 0;
            }
          }
        }
        break;

      case 2:
        this->k_++;
        if (this->k_ >= this->mesh_->min_k_+this->mesh_->nk_-1) {
          this->k_ = this->mesh_->min_k_;
          this->i_++;
          if (this->i_ >=  this->mesh_->min_i_+this->mesh_->ni_) {
            this->i_ = this->mesh_->min_i_;
            this->j_++;
            if (this->j_ >= this->mesh_->min_j_+this->mesh_->nj_) {
              this->dir_++;
              this->i_ = 0;
              this->j_ = 0;
              this->k_ = 0;
            }
          }
        }
        break;
      default:
      case 3:
        ASSERTFAIL("Iterating beyond MaskedLatVolMesh edge boundaries.");
        BREAK;
      }
    }
    void prev() {
      switch(this->dir_)
      {
      case 2:
        if (this->k_ == this->mesh_->min_k_) {
          this->k_ = this->mesh_->min_k_ + this->mesh_->nk_-1;
          if (this->i_ == this->mesh_->min_i_) {
            this->i_ = this->mesh_->min_i_ + this->mesh_->ni_;
            if (this->j_ == this->mesh_->min_j_) {
              this->i_ = this->mesh_->min_i_ + this->mesh_->ni_;
              this->j_ = this->mesh_->min_j_ + this->mesh_->nj_-1;
              this->k_ = this->mesh_->min_k_ + this->mesh_->nk_;
              this->dir_--;
            }
            else {
              this->j_--;
            }
          }
          else {
            this->i_--;
          }
        }
        else {
          this->k_--;
        }
        break;

      case 1:
        if (this->j_ == this->mesh_->min_j_) {
          this->j_ = this->mesh_->min_j_ + this->mesh_->nj_-1;
          if (this->k_ == this->mesh_->min_k_) {
            this->k_ = this->mesh_->min_k_ + this->mesh_->nk_;
            if (this->i_ == this->mesh_->min_i_) {
              this->i_ = this->mesh_->min_i_ + this->mesh_->ni_-1;
              this->j_ = this->mesh_->min_j_ + this->mesh_->nj_;
              this->k_ = this->mesh_->min_k_ + this->mesh_->nk_;
              this->dir_--;
            }
            else {
              this->i_--;
            }
          }
          else {
            this->k_--;
          }
        }
        else {
          this->j_--;
        }
        break;

      case 0:
        if (this->i_ == this->mesh_->min_i_) {
          this->i_ = this->mesh_->min_i_ + this->mesh_->ni_-1;
          if (this->j_ == this->mesh_->min_j_) {
            this->j_ = this->mesh_->min_j_ + this->mesh_->nj_;
            if (this->k_ == this->mesh_->min_k_) {
              ASSERTFAIL("Iterating b4 MaskedLatVolMesh edge boundaries.");
            }
            else {
              this->k_--;
            }
          }
          else {
            this->j_--;
          }
        }
        else {
          this->i_--;
        }
        break;
      default:
        ASSERTFAIL("Iterating beyond MaskedLatVolMesh edge boundaries.");
        BREAK;
      }
    }

  private:
    EdgeIter operator++(int)
    {
      EdgeIter result(*this);
      operator++();
      return result;
    }
    EdgeIter operator--(int)
    {
      EdgeIter result(*this);
      operator--();
      return result;
    }
  };

  struct FaceIter : public FaceIndex
  {
  public:
    FaceIter() : FaceIndex() {}
    FaceIter(const MaskedLatVolMesh *m,
             unsigned int i, unsigned int j, unsigned int k, unsigned int dir)
      : FaceIndex(m, i, j, k, dir){}

    const FaceIndex &operator *() const
    {
      return (const FaceIndex&)(*this);
    }

    FaceIter &operator++() {
      do next(); while (!this->mesh_->check_valid(*this) && this->dir_ < 3);
      return *this;
    }
    FaceIter &operator--() {
      do prev(); while (!this->mesh_->check_valid(*this));
      return *this;
    }

    void next() {
      switch (this->dir_)
      {
      case 0:
        this->i_++;
        if (this->i_ >= this->mesh_->min_i_+this->mesh_->ni_-1) {
          this->i_ = this->mesh_->min_i_;
          this->j_++;
          if (this->j_ >=  this->mesh_->min_j_+this->mesh_->nj_-1) {
            this->j_ = this->mesh_->min_j_;
            this->k_++;
            if (this->k_ >= this->mesh_->min_k_+this->mesh_->nk_) {
              this->dir_++;
              this->i_ = 0;
              this->j_ = 0;
              this->k_ = 0;
            }
          }
        }
        break;
      case 1:
        this->j_++;
        if (this->j_ >= this->mesh_->min_j_+this->mesh_->nj_-1) {
          this->j_ = this->mesh_->min_j_;
          this->k_++;
          if (this->k_ >=  this->mesh_->min_k_+this->mesh_->nk_-1) {
            this->k_ = this->mesh_->min_k_;
            this->i_++;
            if (this->i_ >= this->mesh_->min_i_+this->mesh_->ni_) {
              this->dir_++;
              this->i_ = 0;
              this->j_ = 0;
              this->k_ = 0;
            }
          }
        }
        break;

      case 2:
        this->k_++;
        if (this->k_ >= this->mesh_->min_k_+this->mesh_->nk_-1) {
          this->k_ = this->mesh_->min_k_;
          this->i_++;
          if (this->i_ >=  this->mesh_->min_i_+this->mesh_->ni_-1) {
            this->i_ = this->mesh_->min_i_;
            this->j_++;
            if (this->j_ >= this->mesh_->min_j_+this->mesh_->nj_) {
              this->dir_++;
              this->i_ = 0;
              this->j_ = 0;
              this->k_ = 0;
            }
          }
        }
        break;
      default:
      case 3:
        ASSERTFAIL("Iterating beyond MaskedLatVolMesh edge boundaries.");
        BREAK;
      }
    }
    void prev() {
      switch(this->dir_)
      {
      case 2:
        if (this->k_ == this->mesh_->min_k_) {
          this->k_ = this->mesh_->min_k_ + this->mesh_->nk_-1;
          if (this->i_ == this->mesh_->min_i_) {
            this->i_ = this->mesh_->min_i_ + this->mesh_->ni_-1;
            if (this->j_ == this->mesh_->min_j_) {
              this->i_ = this->mesh_->min_i_ + this->mesh_->ni_;
              this->j_ = this->mesh_->min_j_ + this->mesh_->nj_-1;
              this->k_ = this->mesh_->min_k_ + this->mesh_->nk_-1;
              this->dir_--;
            }
            else {
              this->j_--;
            }
          }
          else {
            this->i_--;
          }
        }
        else {
          this->k_--;
        }
        break;

      case 1:
        if (this->j_ == this->mesh_->min_j_) {
          this->j_ = this->mesh_->min_j_ + this->mesh_->nj_-1;
          if (this->k_ == this->mesh_->min_k_) {
            this->k_ = this->mesh_->min_k_ + this->mesh_->nk_-1;
            if (this->i_ == this->mesh_->min_i_) {
              this->i_ = this->mesh_->min_i_ + this->mesh_->ni_-1;
              this->j_ = this->mesh_->min_j_ + this->mesh_->nj_-1;
              this->k_ = this->mesh_->min_k_ + this->mesh_->nk_;
              this->dir_--;
            }
            else {
              this->i_--;
            }
          }
          else {
            this->k_--;
          }
        }
        else {
          this->j_--;
        }
        break;

      case 0:
        if (this->i_ == this->mesh_->min_i_) {
          this->i_ = this->mesh_->min_i_ + this->mesh_->ni_-1;
          if (this->j_ == this->mesh_->min_j_) {
            this->j_ = this->mesh_->min_j_ + this->mesh_->nj_-1;
            if (this->k_ == this->mesh_->min_k_) {
              ASSERTFAIL("Iterating b4 MaskedLatVolMesh face boundaries.");
            }
            else {
              this->k_--;
            }
          }
          else {
            this->j_--;
          }
        }
        else {
          this->i_--;
        }
        break;
      default:
        ASSERTFAIL("Iterating beyond MaskedLatVolMesh face boundaries.");
        BREAK;
      }
    }

  private:
    FaceIter operator++(int)
    {
      FaceIter result(*this);
      operator++();
      return result;
    }
    FaceIter operator--(int)
    {
      FaceIter result(*this);
      operator--();
      return result;
    }
  };

  struct CellIter : public MaskedLatIndex
  {
  public:
    CellIter() : MaskedLatIndex() {}
    CellIter(const MaskedLatVolMesh *m, unsigned int i,
             unsigned int j, unsigned int k) :
      MaskedLatIndex(m, i, j, k)
    {}

    const CellIndex &operator *() const
    {
      return (const CellIndex&)(*this);
    }

    operator unsigned() const {
      ASSERT(this->mesh_);
      return this->i_ + (this->mesh_->ni_-1)*this->j_ + (this->mesh_->ni_-1)*(this->mesh_->nj_-1)*this->k_;
    }

    CellIter &operator++() {
      do next(); while (!this->mesh_->check_valid(this->i_,this->j_,this->k_) &&
                        this->k_ < this->mesh_->min_k_ + this->mesh_->nk_ - 1);
      return *this;
    }
    CellIter &operator--() {
      do prev(); while (!this->mesh_->check_valid(this->i_,this->j_,this->k_));
      return *this;
    }

    void next() {
      this->i_++;
      if (this->i_ >= this->mesh_->min_i_+this->mesh_->ni_-1)   {
        this->i_ = this->mesh_->min_i_;
        this->j_++;
        if (this->j_ >=  this->mesh_->min_j_+this->mesh_->nj_-1) {
          this->j_ = this->mesh_->min_j_;
          this->k_++;
        }
      }
    }
    void prev() {
      if (this->i_ == this->mesh_->min_i_) {
        this->i_ = this->mesh_->min_i_ + this->mesh_->ni_-1;
        if (this->j_ == this->mesh_->min_j_) {
          this->j_ = this->mesh_->min_j_ + this->mesh_->nj_-1;
          ASSERTMSG(this->k_ != this->mesh_->min_k_, "Cant prev() from first cell!");
          this->k_--;
        }
        else {
          this->j_--;
        }
      }
      else {
        this->i_--;
      }
    }

  private:
    CellIter operator++(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }
    CellIter operator--(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }

  };

  struct Node {
    typedef NodeIndex           index_type;
    typedef NodeIter            iterator;
    typedef NodeSize            size_type;
    typedef vector<index_type>  array_type;
  };

  struct Edge {
    typedef EdgeIndex           index_type;
    typedef EdgeIter            iterator;
    typedef EdgeSize            size_type;
    typedef vector<index_type>  array_type;
  };

  struct Face {
    typedef FaceIndex           index_type;
    typedef FaceIter            iterator;
    typedef FaceSize            size_type;
    typedef vector<index_type>  array_type;
  };

  struct Cell {
    typedef CellIndex           index_type;
    typedef CellIter            iterator;
    typedef CellSize            size_type;
    typedef vector<index_type>  array_type;
  };

  typedef Cell Elem;
  typedef Face DElem;

  friend class NodeIter;
  friend class CellIter;
  friend class EdgeIter;
  friend class FaceIter;

  friend class NodeIndex;
  friend class CellIndex;
  friend class EdgeIndex;
  friend class FaceIndex;

  MaskedLatVolMesh();
  MaskedLatVolMesh(unsigned int x, unsigned int y, unsigned int z,
                   const Point &min, const Point &max);
  MaskedLatVolMesh(const MaskedLatVolMesh &copy);
  virtual MaskedLatVolMesh *clone() { return new MaskedLatVolMesh(*this); }
  virtual ~MaskedLatVolMesh() {}

  Basis &get_basis() { return  LatVolMesh<Basis>::get_basis(); }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename Elem::index_type ci,
                       unsigned which_edge,
                       unsigned div_per_unit) const
  {
    typename LatVolMesh<Basis>::Elem::index_type i(this, ci.i_,
                                                   ci.j_, ci.k_);
    LatVolMesh<Basis>::pwl_approx_edge(coords, i, which_edge, div_per_unit);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename Elem::index_type ci,
                       unsigned which_face,
                       unsigned div_per_unit) const
  {
    typename LatVolMesh<Basis>::Elem::index_type i(this, ci.i_,
                                                   ci.j_, ci.k_);
    LatVolMesh<Basis>::pwl_approx_face(coords, i, which_face, div_per_unit);
  }

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename Elem::index_type idx) const
  {
    typename LatVolMesh<Basis>::Elem::index_type i(this, idx.i_,
                                                   idx.j_, idx.k_);
    return LatVolMesh<Basis>::get_coords(coords, p, i);
  }

  void interpolate(Point &pt, const vector<double> &coords,
                   typename Elem::index_type idx) const
  {
    typename LatVolMesh<Basis>::Elem::index_type i(this, idx.i_,
                                                   idx.j_, idx.k_);
    LatVolMesh<Basis>::interpolate(pt, coords, i);
  }

  // get the Jacobian matrix
  void derivate(const vector<double> &coords,
                typename Elem::index_type idx,
                vector<Point> &J) const
  {
    typename LatVolMesh<Basis>::Elem::index_type i(this, idx.i_,
                                                   idx.j_, idx.k_);
    LatVolMesh<Basis>::derivate(coords, i, J);
  }

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;

  //! Methods specific to MaskedLatVolMesh
  void mask_node(typename Node::index_type);
  void mask_edge(typename Edge::index_type);
  void mask_face(typename Face::index_type);
  void mask_cell(typename Cell::index_type);
  void unmask_node(typename Node::index_type);
  void unmask_edge(typename Edge::index_type);
  void unmask_face(typename Face::index_type);
  void unmask_cell(typename Cell::index_type);
  //! Special Method to Reset Mesh
  void unmask_everything();

  unsigned int num_masked_nodes() const;
  unsigned int num_masked_edges() const;
  unsigned int num_masked_faces() const;
  unsigned int num_masked_cells() const;

  void begin(typename Node::iterator &) const;
  void begin(typename Edge::iterator &) const;
  void begin(typename Face::iterator &) const;
  void begin(typename Cell::iterator &) const;

  void end(typename Node::iterator &) const;
  void end(typename Edge::iterator &) const;
  void end(typename Face::iterator &) const;
  void end(typename Cell::iterator &) const;

  void size(typename Node::size_type &) const;
  void size(typename Edge::size_type &) const;
  void size(typename Face::size_type &) const;
  void size(typename Cell::size_type &) const;

  void to_index(typename Node::index_type &index, unsigned int i);
  void to_index(typename Edge::index_type &index, unsigned int i);
  void to_index(typename Face::index_type &index, unsigned int i);
  void to_index(typename Cell::index_type &index, unsigned int i);

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, typename Face::index_type) const;
  void get_nodes(typename Node::array_type &, typename Cell::index_type) const;
  void get_edges(typename Edge::array_type &, typename Face::index_type) const;
  void get_edges(typename Edge::array_type &,
                 const typename Cell::index_type&) const;
  void get_faces(typename Face::array_type &,
                 const typename Cell::index_type&) const;

  //! get the parent element(s) of the given index
  void get_elems(typename Cell::array_type &result,
                 const typename Node::index_type &idx) const;

  void get_elems(typename Cell::array_type &result,
                 const typename Edge::index_type &idx) const { result.clear(); }

  void get_elems(typename Cell::array_type &result,
                 const typename Face::index_type &idx) const { result.clear(); }


  //! Wrapper to get the derivative elements from this element.
  void get_delems(typename DElem::array_type &result,
                  typename Elem::index_type idx) const
  {
    get_faces(result, idx);
  }

  // returns 26 pairs in ijk order
  void  get_neighbors_stencil(
                              vector<pair<bool,typename Cell::index_type> > &nbrs,
                              typename Cell::index_type idx) const;
  // return 26 pairs in ijk order
  void  get_neighbors_stencil(
                              vector<pair<bool,typename Node::index_type> > &nbrs,
                              typename Node::index_type idx) const;

  // return 8 pairs in ijk order
  void  get_neighbors_stencil(
                              vector<pair<bool,typename Cell::index_type> > &nbrs,
                              typename Node::index_type idx) const;


  //! get the center point (in object space) of an element
  void get_center(Point &, const typename Node::index_type &) const;
  void get_center(Point &, const typename Edge::index_type &) const;
  void get_center(Point &, const typename Face::index_type &) const;
  void get_center(Point &, const typename Cell::index_type &) const;

  double get_size(typename Node::index_type idx) const;
  double get_size(typename Edge::index_type idx) const;
  double get_size(typename Face::index_type idx) const;
  double get_size(typename Cell::index_type idx) const;
  double get_length(typename Edge::index_type idx) const
  { return get_size(idx); };
  double get_area(typename Face::index_type idx) const
  { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const
  { return get_size(idx); };

  bool locate(typename Node::index_type &, const Point &);
  bool locate(typename Edge::index_type &, const Point &) const
  { return false; }
  bool locate(typename Face::index_type &, const Point &) const
  { return false; }
  bool locate(typename Cell::index_type &, const Point &);

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * )
  {ASSERTFAIL("LatVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , typename Face::array_type & , double * )
  {ASSERTFAIL("LatVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, typename Cell::array_type &l, double *w);

  void get_point(Point &point, const typename Node::index_type &index) const
  { get_center(point, index); }

  void get_normal(Vector &/*normal*/,
                  typename Node::index_type /*index*/) const
  { ASSERTFAIL("This mesh type does not have node normals."); }
  void get_normal(Vector &, vector<double> &, typename Elem::index_type,
                  unsigned int)
  { ASSERTFAIL("This mesh type does not have element normals."); }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);

  virtual const TypeDescription *get_type_description() const;
  static const TypeDescription* cell_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* node_type_description();
  static const TypeDescription* elem_type_description()
  { return cell_type_description(); }
  static const TypeDescription* cell_index_type_description();
  static const TypeDescription* node_index_type_description();

  unsigned int  get_sequential_node_index(const typename Node::index_type idx);

  // returns a MaskedLatVolMesh
  static Persistent *maker() { return new MaskedLatVolMesh<Basis>(); }

private:
  unsigned int  synchronized_;
  map<typename Node::index_type, unsigned>      nodes_;
  Mutex                                 node_lock_;
  set<unsigned int> masked_cells_;
  unsigned          masked_nodes_count_;
  unsigned          masked_edges_count_;
  unsigned          masked_faces_count_;

  bool          update_count(typename Cell::index_type, bool masking);
  unsigned      num_missing_faces(typename Cell::index_type);
  bool          check_valid(typename Node::index_type idx) const;
  bool          check_valid(typename Edge::index_type idx) const;
  bool          check_valid(typename Face::index_type idx) const;
  bool          check_valid(typename Cell::index_type idx) const;
  bool          check_valid(typename Node::iterator idx) const;
  bool          check_valid(typename Edge::iterator idx) const;
  bool          check_valid(typename Face::iterator idx) const;
  bool          check_valid(typename Cell::iterator idx) const;

  inline bool   check_valid(int i, int j, int k) const
  {
    if ((i >= int(this->min_i_)) && (i <(int(this->min_i_ +this->ni_) - 1)) &&
        (j >= int(this->min_j_)) && (j <(int(this->min_j_ +this->nj_) - 1)) &&
        (k >= int(this->min_k_)) && (k <(int(this->min_k_ +this->nk_) - 1)) &&
        (masked_cells_.find(unsigned(typename Cell::index_type(this,i,j,k))) == masked_cells_.end()))
    {
      return true;
    }
    return false;
  }
};


template <class Basis>
PersistentTypeID
MaskedLatVolMesh<Basis>::type_id(type_name(-1),
                                 LatVolMesh<Basis>::type_name(-1),
                                 maker);

template <class Basis>
MaskedLatVolMesh<Basis>::MaskedLatVolMesh():
  LatVolMesh<Basis>(),
  synchronized_(0),
  nodes_(),
  node_lock_("MaskedLatVolMesh node_lock_"),
  masked_cells_(),
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)
{
}


template <class Basis>
MaskedLatVolMesh<Basis>::MaskedLatVolMesh(unsigned int x,
                                          unsigned int y,
                                          unsigned int z,
                                          const Point &min,
                                          const Point &max) :
  LatVolMesh<Basis>(x, y, z, min, max),
  synchronized_(0),
  nodes_(),
  node_lock_("MaskedLatVolMesh node_lock_"),
  masked_cells_(),
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)
{
}


template <class Basis>
MaskedLatVolMesh<Basis>::MaskedLatVolMesh(const MaskedLatVolMesh<Basis> &copy):
  LatVolMesh<Basis>(copy),
  synchronized_(copy.synchronized_),
  nodes_(copy.nodes_),
  node_lock_("MaskedLatVolMesh node_lock_"),
  masked_cells_(copy.masked_cells_),
  masked_nodes_count_(copy.masked_nodes_count_),
  masked_edges_count_(copy.masked_edges_count_),
  masked_faces_count_(copy.masked_edges_count_)
{
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Node::iterator &itr) const
{
  itr = typename Node::iterator(this,
                                this->min_i_, this->min_j_, this->min_k_);
  if (!check_valid(itr)) ++itr;
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Node::iterator &itr) const
{
  itr = typename Node::iterator(this, this->min_i_, this->min_j_, this->min_k_ + this->nk_);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Node::size_type &s) const
{
  s = typename Node::size_type(this,this->ni_,this->nj_,this->nk_);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Node::index_type &idx,
                                  unsigned int a)
{
  const unsigned int i = a % this->ni_;
  const unsigned int jk = a / this->ni_;
  const unsigned int j = jk % this->nj_;
  const unsigned int k = jk / this->nj_;
  idx = typename Node::index_type(this, i, j, k);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(this,  this->min_i_, this->min_j_, this->min_k_);
  if (!check_valid(itr)) ++itr;
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(this, this->min_i_, this->min_j_, this->min_k_ + this->nk_-1);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Cell::size_type &s) const
{
  s = typename Cell::size_type(this,this->ni_-1, this->nj_-1,this->nk_-1);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Cell::index_type &idx,
                                  unsigned int a)
{
  const unsigned int i = a % (this->ni_-1);
  const unsigned int jk = a / (this->ni_-1);
  const unsigned int j = jk % (this->nj_-1);
  const unsigned int k = jk / (this->nj_-1);
  idx = typename Cell::index_type(this, i, j, k);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Edge::iterator &itr) const
{
  itr = typename Edge::iterator(this,this->min_i_,this->min_j_,this->min_k_,0);
  if (!check_valid(itr)) ++itr;
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Edge::iterator &itr) const
{
  itr = typename Edge::iterator(this, this->min_i_, this->min_j_, this->min_k_,3);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Edge::size_type &s) const
{
  s = typename Edge::size_type(this,this->ni_,this->nj_,this->nk_);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Edge::index_type &idx,
                                  unsigned int a)
{
  idx = a;
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Face::iterator &itr) const
{
  itr = typename Face::iterator(this,this->min_i_,this->min_j_,this->min_k_,0);
  if (!check_valid(itr)) ++itr;
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Face::iterator &itr) const
{
  itr = typename Face::iterator(this, this->min_i_, this->min_j_, this->min_k_,3);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Face::size_type &s) const
{
  s = typename Face::size_type(this,this->ni_,this->nj_,this->nk_);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Face::index_type &idx,
                                  unsigned int a)
{
  idx = a;
}


//! get the child elements of the given index
template <class Basis>
void
MaskedLatVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Edge::index_type e) const
{
  array.resize(2);
  array[0] = typename Node::index_type(this,e.i_,e.j_,e.k_);
  array[1] = typename Node::index_type(this,
                                       e.i_ + (e.dir_ == 0 ? 1:0),
                                       e.j_ + (e.dir_ == 1 ? 1:0),
                                       e.k_ + (e.dir_ == 2 ? 1:0));
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Face::index_type f) const
{
  array.resize(4);
  array[0] = typename Node::index_type(this,f.i_,f.j_,f.k_);
  array[1] = typename Node::index_type(this,
                                       f.i_ + (f.dir_ == 0 ? 1:0),
                                       f.j_ + (f.dir_ == 1 ? 1:0),
                                       f.k_ + (f.dir_ == 2 ? 1:0));
  array[2] = typename Node::index_type(this,
                                       f.i_ + ((f.dir_ == 0 || f.dir_ == 2) ? 1:0),
                                       f.j_ + ((f.dir_ == 0 || f.dir_ == 1) ? 1:0),
                                       f.k_ + ((f.dir_ == 1 || f.dir_ == 2) ? 1:0));
  array[3] = typename Node::index_type(this,
                                       f.i_ + (f.dir_ == 2 ? 1:0),
                                       f.j_ + (f.dir_ == 0 ? 1:0),
                                       f.k_ + (f.dir_ == 1 ? 1:0));
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Cell::index_type idx) const
{
  array.resize(8);
  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;   array[0].k_ = idx.k_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;   array[1].k_ = idx.k_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1; array[2].k_ = idx.k_;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1; array[3].k_ = idx.k_;
  array[4].i_ = idx.i_;   array[4].j_ = idx.j_;   array[4].k_ = idx.k_+1;
  array[5].i_ = idx.i_+1; array[5].j_ = idx.j_;   array[5].k_ = idx.k_+1;
  array[6].i_ = idx.i_+1; array[6].j_ = idx.j_+1; array[6].k_ = idx.k_+1;
  array[7].i_ = idx.i_;   array[7].j_ = idx.j_+1; array[7].k_ = idx.k_+1;
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_edges(typename Edge::array_type &array,
                                   typename Face::index_type idx) const
{
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_edges(typename Edge::array_type &array,
                                   const typename Cell::index_type &idx) const
{
  array.resize(12);
  array[0] = EdgeIndex(this, idx.i_, idx.j_, idx.k_, 0);
  array[1] = EdgeIndex(this, idx.i_, idx.j_+ 1, idx.k_, 0);
  array[2] = EdgeIndex(this, idx.i_, idx.j_, idx.k_ + 1, 0);
  array[3] = EdgeIndex(this, idx.i_, idx.j_ + 1, idx.k_ + 1, 0);

  array[4] = EdgeIndex(this, idx.i_, idx.j_, idx.k_, 1);
  array[5] = EdgeIndex(this, idx.i_ + 1, idx.j_, idx.k_, 1);
  array[6] = EdgeIndex(this, idx.i_, idx.j_, idx.k_ + 1, 1);
  array[7] = EdgeIndex(this, idx.i_ + 1, idx.j_, idx.k_ + 1, 1);

  array[8] =  EdgeIndex(this, idx.i_, idx.j_, idx.k_, 2);
  array[9] =  EdgeIndex(this, idx.i_ + 1, idx.j_, idx.k_, 2);
  array[10] = EdgeIndex(this, idx.i_, idx.j_ + 1, idx.k_, 2);
  array[11] = EdgeIndex(this, idx.i_ + 1, idx.j_ + 1, idx.k_, 2);

}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_faces(typename Face::array_type &array,
                                   const typename Cell::index_type &idx) const
{
  array.resize(6);
  const unsigned int i = idx.i_;
  const unsigned int j = idx.j_;
  const unsigned int k = idx.k_;

  array[0] = FaceIndex(this, i, j, k, 0);
  array[1] = FaceIndex(this, i, j, k+1, 0);

  array[2] = FaceIndex(this, i, j, k, 1);
  array[3] = FaceIndex(this, i+1, j, k, 1);

  array[4] = FaceIndex(this, i, j, k, 2);
  array[5] = FaceIndex(this, i, j+1, k, 2);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_elems(
          typename MaskedLatVolMesh<Basis>::Cell::array_type &result,
          const typename MaskedLatVolMesh<Basis>::Node::index_type &idx) const
{
  result.reserve(8);
  result.clear();
  const unsigned int i0 = idx.i_ ? idx.i_ - 1 : 0;
  const unsigned int j0 = idx.j_ ? idx.j_ - 1 : 0;
  const unsigned int k0 = idx.k_ ? idx.k_ - 1 : 0;

  const unsigned int i1 = idx.i_ < this->ni_-1 ? idx.i_+1 : this->ni_-1;
  const unsigned int j1 = idx.j_ < this->nj_-1 ? idx.j_+1 : this->nj_-1;
  const unsigned int k1 = idx.k_ < this->nk_-1 ? idx.k_+1 : this->nk_-1;

  unsigned int i, j, k;
  for (k = k0; k < k1; k++)
    for (j = j0; j < j1; j++)
      for (i = i0; i < i1; i++)
        if (check_valid(i, j, k))
          result.push_back(typename Cell::index_type(this, i, j, k));
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Node::index_type idx) const
{
  unsigned int i = idx.i_, j = idx.j_, k = idx.k_;
  return (check_valid(i  ,j  ,k  ) ||
          check_valid(i-1,j  ,k  ) ||
          check_valid(i  ,j-1,k  ) ||
          check_valid(i  ,j  ,k-1) ||
          check_valid(i-1,j-1,k  ) ||
          check_valid(i-1,j  ,k-1) ||
          check_valid(i  ,j-1,k-1) ||
          check_valid(i-1,j-1,k-1));
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(
                 typename MaskedLatVolMesh<Basis>::Edge::index_type idx) const
{

  bool val = false;
  if (idx.dir_ == 0)
  {
    val =  ((idx.i_ < this->min_i_ + this->ni_ - 1) &&
            (check_valid(idx.i_  ,idx.j_  ,idx.k_  ) ||
             check_valid(idx.i_  ,idx.j_-1,idx.k_  ) ||
             check_valid(idx.i_  ,idx.j_  ,idx.k_-1) ||
             check_valid(idx.i_  ,idx.j_-1,idx.k_-1)));
  }
  if (idx.dir_ == 1)
  {
    val =   ((idx.j_ < this->min_j_ + this->nj_ - 1) &&
             (check_valid(idx.i_  ,idx.j_  ,idx.k_) ||
              check_valid(idx.i_-1,idx.j_  ,idx.k_) ||
              check_valid(idx.i_  ,idx.j_  ,idx.k_-1) ||
              check_valid(idx.i_-1,idx.j_  ,idx.k_-1)));
  }
  if (idx.dir_ == 2)
  {
    val =  ((idx.k_ < this->min_k_ + this->nk_ - 1) &&
            (check_valid(idx.i_  ,idx.j_,  idx.k_) ||
             check_valid(idx.i_-1,idx.j_,  idx.k_) ||
             check_valid(idx.i_  ,idx.j_-1,idx.k_) ||
             check_valid(idx.i_-1,idx.j_-1,idx.k_)));
  }
  return val;
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(
                  typename MaskedLatVolMesh<Basis>::Face::index_type idx) const
{
  if (idx.dir_ == 0)
  {
    return (idx.i_ < this->min_i_ + this->ni_ - 1 &&
            idx.j_ < this->min_j_ + this->nj_ - 1 &&
            (check_valid(idx.i_,idx.j_,idx.k_) ||
             check_valid(idx.i_,idx.j_,idx.k_-1)));
  }
  if (idx.dir_ == 1)
  {
    return (idx.j_ < this->min_j_ + this->nj_ - 1 &&
            idx.k_ < this->min_k_ + this->nk_ - 1 &&
            (check_valid(idx.i_,idx.j_,idx.k_) ||
             check_valid(idx.i_-1,idx.j_,idx.k_)));
  }
  if (idx.dir_ == 2)
  {
    return (idx.i_ < this->min_i_ + this->ni_ - 1 &&
            idx.k_ < this->min_k_ + this->nk_ - 1 &&
            (check_valid(idx.i_,idx.j_,idx.k_) ||
             check_valid(idx.i_,idx.j_-1,idx.k_)));
  }

  return false;
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(
                  typename MaskedLatVolMesh<Basis>::Cell::index_type i) const
{
  return check_valid(i.i_,i.j_,i.k_);
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(
                  typename MaskedLatVolMesh<Basis>::Node::iterator i) const
{
  return check_valid(*i);
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(
                  typename MaskedLatVolMesh<Basis>::Edge::iterator i) const
{
  return check_valid(*i);
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(
                  typename MaskedLatVolMesh<Basis>::Face::iterator i) const
{
  return check_valid(*i);
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(
                  typename MaskedLatVolMesh<Basis>::Cell::iterator i) const
{
  return check_valid(i.i_,i.j_,i.k_);
}


//! This function updates the missing node, edge, and face count
//! when masking or unmasking one cell.
//! Returns true if nodes, edges, or faces count is changed
template <class Basis>
bool
MaskedLatVolMesh<Basis>::update_count(typename MaskedLatVolMesh<Basis>::Cell::index_type c,
                                      bool masking)
{
  synchronized_ &= ~Mesh::NODES_E;
  const bool i0 = (c.i_ > this->min_i_) && check_valid(c.i_-1, c.j_, c.k_);
  const bool j0 = (c.j_ > this->min_j_) && check_valid(c.i_, c.j_-1, c.k_);
  const bool k0 = (c.k_ > this->min_k_) && check_valid(c.i_, c.j_, c.k_-1);
  const bool i1 = (c.i_ < this->min_i_+this->ni_-1) && check_valid(c.i_+1, c.j_, c.k_);
  const bool j1 = (c.j_ < this->min_j_+this->nj_-1) && check_valid(c.i_, c.j_+1, c.k_);
  const bool k1 = (c.k_ < this->min_k_+this->nk_-1) && check_valid(c.i_, c.j_, c.k_+1);

  // These counts are the number of nodes, edges, faces that exist
  // ONLY from the presence of this cell, not because of the contribution
  // of neighboring cells.
  const unsigned int faces = (i0?0:1)+(i1?0:1)+(j0?0:1)+(j1?0:1)+(k0?0:1)+(k1?0:1);
  unsigned int       nodes = 0;
  unsigned int       edges = 0;

  if (faces == 6) {
    nodes = 8;
    edges = 12;
  }
  else {
    if (faces == 5)     {
      nodes = 4; edges = 8;
    }
    else {
      if (faces == 1 || faces == 0)     {
        nodes = 0; edges = 0;
      }
      else {
        if(faces == 4) {
          if((i0 == i1) && (j0 == j1) && (k0 == k1)) {
            nodes = 0;
            edges = 4;
          }
          else {
            nodes = 2;
            edges = 5;
          }
        }
        else {
          if(faces == 3) {
            if((i0!=i1)&&(j0!=j1)&&(k0!=k1)) {
              nodes = 1;
              edges = 3;
            }
            else {
              nodes = 0;
              nodes = 2;
            }
          }
          else {
            if(faces == 2) {
              if((i0 == i1) && (j0 == j1) && (k0 == k1)) {
                nodes = 0;
                edges = 0;
              }
              else {
                nodes = 0;
                edges = 1;
              }
            }
          }
        }
      }
    }
  }

  // These nodes, edges, faces are being implicitly removed from the mesh
  // by the removal of this cell.
  if (masking)
  {
    masked_nodes_count_ += nodes;
    masked_edges_count_ += edges;
    masked_faces_count_ += faces;
  }
  // These ndoes, edges, & faces are being implicitly added back into the mesh
  // because this cell is being added back in
  else
  {
    masked_nodes_count_ -= nodes;
    masked_edges_count_ -= edges;
    masked_faces_count_ -= faces;
  }

  return (faces == 0);
}


template <class Basis>
BBox
MaskedLatVolMesh<Basis>::get_bounding_box() const
{
  // TODO:  return bounding box of valid cells only
  return LatVolMesh<Basis>::get_bounding_box();
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Node::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center
    (result, typename LatVolMesh<Basis>::Node::index_type(this,idx.i_,idx.j_,idx.k_));
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Edge::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center(result, typename LatVolMesh<Basis>::Edge::index_type(unsigned(idx)));
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Face::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center(result, typename LatVolMesh<Basis>::Face::index_type(unsigned(idx)));
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Cell::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center
    (result,typename LatVolMesh<Basis>::Cell::index_type(this,idx.i_,idx.j_,idx.k_));
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::locate(typename Node::index_type &idx, const Point &p)
{
  typename LatVolMesh<Basis>::Node::index_type i(this,idx.i_,idx.j_,idx.k_);
  bool lat = LatVolMesh<Basis>::locate(i,p);
  idx.i_ = i.i_; idx.j_ = i.j_; idx.k_ = i.k_; idx.mesh_ = this;
  if (lat && check_valid(idx))
  {
    return true;
  }
  else return false;
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::locate(typename Cell::index_type &idx, const Point &p)
{
  if (this->basis_.polynomial_order() > 1) {
    if (elem_locate(idx, *this, p) && check_valid(idx)) return true;
  }
  typename LatVolMesh<Basis>::Cell::index_type i(this,idx.i_,idx.j_,idx.k_);
  bool lat = LatVolMesh<Basis>::locate(i,p);
  idx.i_ = i.i_; idx.j_ = i.j_; idx.k_ = i.k_; idx.mesh_ = this;
  if (lat && check_valid(idx))
  {
    return true;
  }
  else return false;
}


template <class Basis>
int
MaskedLatVolMesh<Basis>::get_weights(const Point &p,
                                     typename Node::array_type &l,
                                     double *w)
{
  return LatVolMesh<Basis>::get_weights(p, l, w);
}


template <class Basis>
int
MaskedLatVolMesh<Basis>::get_weights(const Point &p,
                                     typename Cell::array_type &l,
                                     double *w)
{
  return LatVolMesh<Basis>::get_weights(p, l, w);
}


template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Node::index_type idx) const
{
  ASSERT(check_valid(idx));
  typename LatVolMesh<Basis>::Node::index_type i(this,idx.i_,idx.j_,idx.k_);
  return LatVolMesh<Basis>::get_size(i);
}


template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Edge::index_type idx) const
{
  ASSERT(check_valid(idx));
  return LatVolMesh<Basis>::get_size(typename LatVolMesh<Basis>::Edge::index_type(unsigned(idx)));
}


template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Face::index_type idx) const
{
  ASSERT(check_valid(idx));
  return LatVolMesh<Basis>::get_size(typename LatVolMesh<Basis>::Face::index_type(unsigned(idx)));
}


template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Cell::index_type idx) const
{
  ASSERT(check_valid(idx));
  typename LatVolMesh<Basis>::Cell::index_type i(this,idx.i_,idx.j_,idx.k_);
  return LatVolMesh<Basis>::get_size(i);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::mask_cell(typename Cell::index_type idx)
{
  update_count(idx,true);
  masked_cells_.insert(unsigned(idx));
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::unmask_cell(typename Cell::index_type idx)
{
  update_count(idx,false);
  masked_cells_.erase(unsigned(idx));
}


template <class Basis>
unsigned int
MaskedLatVolMesh<Basis>::num_masked_nodes() const
{
  return masked_nodes_count_;
}


template <class Basis>
unsigned int
MaskedLatVolMesh<Basis>::num_masked_edges() const
{
  return masked_edges_count_;
}


template <class Basis>
unsigned int
MaskedLatVolMesh<Basis>::num_masked_faces() const
{
  return masked_faces_count_;
}


template <class Basis>
unsigned int
MaskedLatVolMesh<Basis>::num_masked_cells() const
{
  return masked_cells_.size();
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::
get_neighbors_stencil(vector<pair<bool,typename Cell::index_type> > &nbrs,
                      typename Cell::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_ + 1); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_ + 1); j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_ + 1); i++)
        if (i != int(idx.i_) || j != int(idx.j_) || k != int(idx.k_)) {
          if (i >= int(this->min_i_) && j >= int(this->min_j_) && k >= int(this->min_k_) &&
              i <= int(this->min_i_+this->ni_)-1 && j <= int(this->min_j_+this->nj_)-1 &&
              i <= int(this->min_k_+this->nk_)-1 && check_valid(i,j,k)) {
            nbrs.push_back(make_pair(true,typename Cell::index_type(this,i,j,k)));
          } else
            nbrs.push_back(make_pair(false,typename Cell::index_type(0,0,0,0)));
        }
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::
get_neighbors_stencil(vector<pair<bool,typename Node::index_type> > &nbrs,
                      typename Node::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_) + 1; k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_) + 1; j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_) + 1; i++)
        if (i != int(idx.i_) || j != int(idx.j_) || k != int(idx.k_))
        {
          if (i >= int(this->min_i_) && j >= int(this->min_j_) && k >= int(this->min_k_) &&
              i <= int(this->min_i_+this->ni_) && j <= int(this->min_j_+this->nj_) &&
              i <= int(this->min_k_+this->nk_) &&
              check_valid(typename Node::index_type(this,i,j,k)))
            nbrs.push_back(make_pair(true,typename Node::index_type(this,i,j,k)));
          else
            nbrs.push_back(make_pair(false,typename Node::index_type(0,0,0,0)));
        }
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::
get_neighbors_stencil(vector<pair<bool,typename Cell::index_type> > &nbrs,
                      typename Node::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_); j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_); i++)
        if (i >= int(this->min_i_) && j >=
            int(this->min_j_) && k >= int(this->min_k_) &&
            i <= int(this->min_i_+this->ni_)-1 && j <=
            int(this->min_j_+this->nj_)-1 &&
            i <= int(this->min_k_+this->nk_)-1 && check_valid(i,j,k))
          nbrs.push_back(make_pair(true,
                                   typename Cell::index_type(this,i,j,k)));
        else
          nbrs.push_back(make_pair(false,typename Cell::index_type(0,0,0,0)));
}


template <class Basis>
unsigned int
MaskedLatVolMesh<Basis>::get_sequential_node_index(const typename Node::index_type idx)
{
  node_lock_.lock();
  if (synchronized_ & Mesh::NODES_E) {
    node_lock_.unlock();
  }

  nodes_.clear();
  int i = 0;
  typename Node::iterator node, nend;
  begin(node);
  end(nend);
  while (node != nend) {
    nodes_[*node] = i++;
    ++node;
  }
  synchronized_ |= Mesh::NODES_E;
  node_lock_.unlock();

  return nodes_[idx];
}




template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::NodeIndex& n)
{
  stream.begin_cheap_delim();
  Pio(stream, n.i_);
  Pio(stream, n.j_);
  Pio(stream, n.k_);
  stream.end_cheap_delim();
}


template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::CellIndex& n)
{
  stream.begin_cheap_delim();
  Pio(stream, n.i_);
  Pio(stream, n.j_);
  Pio(stream, n.k_);
  stream.end_cheap_delim();
}


template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::EdgeIndex& n)
{
  stream.begin_cheap_delim();
  Pio(stream, n.i_);
  Pio(stream, n.j_);
  Pio(stream, n.k_);
  stream.end_cheap_delim();
}


template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::FaceIndex& n)
{
  stream.begin_cheap_delim();
  Pio(stream, n.i_);
  Pio(stream, n.j_);
  Pio(stream, n.k_);
  stream.end_cheap_delim();
}


template <class Basis>
const string
find_type_name(typename MaskedLatVolMesh<Basis>::NodeIndex *)
{
  static string name = MaskedLatVolMesh<Basis>::type_name(-1) + "::NodeIndex";
  return name;
}


template <class Basis>
const string
find_type_name(typename MaskedLatVolMesh<Basis>::CellIndex *)
{
  static string name = MaskedLatVolMesh<Basis>::type_name(-1) + "::CellIndex";
  return name;
}


#define MASKED_LAT_VOL_MESH_VERSION 1

template <class Basis>
void
MaskedLatVolMesh<Basis>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), MASKED_LAT_VOL_MESH_VERSION);

  LatVolMesh<Basis>::io(stream);

  // IO data members, in order
  vector<unsigned int > masked_vec(masked_cells_.begin(),
                                   masked_cells_.end());
  Pio(stream, masked_vec);
  if (stream.reading())
  {
    masked_cells_.clear();
    masked_cells_.insert(masked_vec.begin(), masked_vec.end());
  }

  stream.end_class();
}


template <class Basis>
const string
MaskedLatVolMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("MaskedLatVolMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
get_type_description(MaskedLatVolMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("MaskedLatVolMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::node_index_type_description()
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *me =
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::NodeIndex",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::cell_index_type_description()
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *me =
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::CellIndex",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


} // namespace SCIRun

#endif // SCI_project_MaskedLatVolMesh_h
