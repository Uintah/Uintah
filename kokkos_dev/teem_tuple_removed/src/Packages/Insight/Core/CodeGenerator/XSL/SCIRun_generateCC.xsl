<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
<xsl:output method="text" indent="yes"/>



<!-- ============= GLOBALS =============== -->
<xsl:variable name="root-name">
  <xsl:value-of select="/filter/@name"/>
</xsl:variable>
<xsl:variable name="sci-name">
  <xsl:value-of select="/filter/filter-sci/@name"/>
</xsl:variable>
<xsl:variable name="itk-name">
  <xsl:value-of select="/filter/filter-itk/@name"/>
</xsl:variable>
<xsl:variable name="package">
  <xsl:value-of select="/filter/filter-sci/package"/>
</xsl:variable>
<xsl:variable name="category">
  <xsl:value-of select="/filter/filter-sci/category"/>
</xsl:variable>
<!-- Variable has_defined_objects indicates whethter this filter
     has defined objects using the datatype tag.  If this is the
     case, we need to add a dimension variable and set a window
     in size. 
-->
<xsl:variable name="has_defined_objects"><xsl:value-of select="/filter/filter-itk/datatypes"/></xsl:variable>








<!-- ================ FILTER ============= -->
<xsl:template match="/filter">
<xsl:call-template name="header" />
<xsl:call-template name="includes" />
namespace <xsl:value-of select="$package"/> 
{

using namespace SCIRun;

<xsl:call-template name="create_class_decl" />

<xsl:call-template name="define_run" />

DECLARE_MAKER(<xsl:value-of select="/filter/filter-sci/@name"/>)

<xsl:call-template name="define_constructor" />
<xsl:call-template name="define_destructor" />
<xsl:call-template name="define_execute" />
<xsl:call-template name="define_processevent" />
<xsl:call-template name="define_constprocessevent" />
<xsl:if test="/filter/filter-sci/outputs/output">
<xsl:call-template name="define_update_after_iteration" />
</xsl:if>
<xsl:call-template name="define_do_its" />
<xsl:call-template name="define_observe" />
<xsl:call-template name="define_tcl_command" />
} // End of namespace Insight
</xsl:template>



<!-- ============= HELPER FUNCTIONS ============ -->

<!-- ====== HEADER ====== -->
<xsl:template name="header">
<xsl:call-template name="output_copyright" />
<xsl:text>/*
 * </xsl:text><xsl:value-of select="$sci-name"/><xsl:text>.cc
 *
 *   Auto Generated File For </xsl:text><xsl:value-of select="$itk-name"/><xsl:text>
 *
 */

</xsl:text>
</xsl:template>


<!-- ======= OUTPUT_COPYRIGHT ====== -->
<xsl:template name="output_copyright">
<xsl:text>/*
  The contents of this file are subject to the University of Utah Public
  License (the &quot;License&quot;); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an &quot;AS IS&quot;
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

</xsl:text>
</xsl:template>

<!-- ====== INCLUDES ====== -->
<xsl:template name="includes">
#include &lt;Dataflow/Network/Module.h&gt;
#include &lt;Core/Malloc/Allocator.h&gt;
#include &lt;Core/GuiInterface/GuiVar.h&gt;
#include &lt;Packages/Insight/share/share.h&gt;
#include &lt;Packages/Insight/Dataflow/Ports/ITKDatatypePort.h&gt;
<xsl:for-each select="/filter/filter-sci/includes/file">
#include &lt;<xsl:value-of select="."/>&gt;
</xsl:for-each>
<xsl:for-each select="/filter/filter-itk/includes/file">
#include &lt;<xsl:value-of select="."/>&gt;
</xsl:for-each>
#include &lt;itkCommand.h&gt;
</xsl:template>



<!-- ====== CREATE_CLASS_DECL ====== -->
<xsl:template name="create_class_decl">
<xsl:text>class </xsl:text><xsl:value-of select="$package"/><xsl:text>SHARE </xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text> : public Module 
{
public:
</xsl:text>
  typedef itk::MemberCommand&lt; <xsl:value-of select="/filter/filter-sci/@name"/> &gt; RedrawCommandType;
<xsl:text>
  // Filter Declaration
  itk::Object::Pointer filter_;

  // Declare GuiVars
  </xsl:text>
<xsl:call-template name="declare_guivars"/>
<xsl:text>  

  // Declare Ports
  </xsl:text>
<xsl:call-template name="declare_ports_and_handles" />
<xsl:text>
  </xsl:text>
<xsl:value-of select="$sci-name"/>
<xsl:text>(GuiContext*);

  virtual ~</xsl:text><xsl:value-of select="$sci-name"/>
<xsl:text disable-output-escaping="yes">();

  virtual void execute();

  virtual void tcl_command(GuiArgs&amp;, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template&lt;</xsl:text>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:choose>
   <xsl:when test="@type"><xsl:value-of select="@type"/><xsl:text> </xsl:text> </xsl:when>
   <xsl:otherwise>class </xsl:otherwise>
</xsl:choose> 
<xsl:value-of select="."/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:for-each><xsl:text> &gt; 
  bool run( </xsl:text><xsl:for-each select="/filter/filter-itk/inputs/input">itk::Object* <xsl:text> </xsl:text><xsl:if test="position() &lt; last()"><xsl:text>, </xsl:text></xsl:if></xsl:for-each> );

  // progress bar
<xsl:if test="/filter/filter-sci/outputs/output">
  void update_after_iteration();
</xsl:if>
  void ProcessEvent(itk::Object * caller, const itk::EventObject &amp; event );
  void ConstProcessEvent(const itk::Object * caller, const itk::EventObject &amp; event );
  void Observe( itk::Object *caller );
  RedrawCommandType::Pointer m_RedrawCommand;
<xsl:for-each select="/filter/filter-sci/outputs/output">
<xsl:variable name="send"><xsl:value-of select="@send_intermediate"/></xsl:variable>
<xsl:if test="$send='yes'"><xsl:text>
  template&lt;</xsl:text>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:choose>
   <xsl:when test="@type"><xsl:value-of select="@type"/><xsl:text> </xsl:text> </xsl:when>
   <xsl:otherwise>class </xsl:otherwise>
</xsl:choose> 
<xsl:value-of select="."/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:for-each>&gt;
  bool do_it_<xsl:value-of select="@name"/>();
  unsigned int iterationCounter_<xsl:value-of select="@name"/>;
</xsl:if>
</xsl:for-each>
};

</xsl:template>



<!-- ====== DECLARE_GUIVARS ====== -->
<xsl:template name="declare_guivars">
<xsl:for-each select="/filter/filter-itk/parameters/param">
<xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="const"><xsl:call-template name="determine_if_const_parameter"/></xsl:variable>

<xsl:variable name="defined_object">
<xsl:call-template name="determine_type"/>
</xsl:variable>
<xsl:if test="$const!='yes'">
<xsl:choose>
<xsl:when test="$defined_object = 'yes'">
<xsl:variable name="data_type"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="type2"><xsl:value-of select="/filter/filter-itk/datatypes/array[@name=$data_type]/elem-type"/></xsl:variable>

<xsl:choose>
<xsl:when test="$type2 = 'double'">vector&lt; GuiDouble* &gt; </xsl:when>
<xsl:when test="$type2 = 'float'">vector&lt; GuiDouble* &gt; </xsl:when>
<xsl:when test="$type2 = 'int'">vector&lt; GuiInt* &gt; </xsl:when>
<xsl:when test="$type2 = 'bool'">vector&lt; GuiInt* &gt; </xsl:when>
<xsl:otherwise>
vector&lt; GuiDouble* &gt; </xsl:otherwise>
</xsl:choose>
</xsl:when>
<xsl:otherwise>

<xsl:choose>
<xsl:when test="$type = 'double'">GuiDouble </xsl:when>
<xsl:when test="$type = 'float'">GuiDouble </xsl:when>
<xsl:when test="$type = 'int'">GuiInt </xsl:when>
<xsl:when test="$type = 'bool'">GuiInt </xsl:when>
<xsl:when test="$type = 'char'">GuiString </xsl:when>
<xsl:when test="$type = 'unsigned char'">GuiString </xsl:when>
<xsl:when test="$type = 'short'">GuiInt </xsl:when>
<xsl:when test="$type = 'unsigned short'">GuiInt </xsl:when>
<xsl:otherwise>
GuiDouble </xsl:otherwise>
</xsl:choose>
</xsl:otherwise>
</xsl:choose>
<xsl:text> gui_</xsl:text><xsl:value-of select="name"/>
<xsl:text>_;
  </xsl:text></xsl:if>
</xsl:for-each>
<xsl:for-each select="/filter/filter-sci/outputs/output">
<xsl:variable name="send"><xsl:value-of select="@send_intermediate"/></xsl:variable>
<xsl:if test="$send='yes'">
  GuiInt gui_update_<xsl:value-of select="@name"/>_;
  GuiInt gui_update_iters_<xsl:value-of select="@name"/>_;
</xsl:if>
</xsl:for-each>
<xsl:if test="$has_defined_objects != ''">
  GuiInt gui_dimension_;</xsl:if>
  bool execute_;
</xsl:template>




<!-- ====== DECLARE_PORTS_AND_HANDLES ====== -->
<xsl:template name="declare_ports_and_handles">

<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="type-name"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="iport">inport_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="ihandle">inhandle_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>
<!-- hard coded datatype --><xsl:text>ITKDatatypeIPort* </xsl:text><xsl:value-of select="$iport"/><xsl:text>_;
  </xsl:text><!-- hard coded datatype --><xsl:text>ITKDatatypeHandle </xsl:text><xsl:value-of select="$ihandle"/>_;
  int last_<xsl:value-of select="@name"/>_;
<xsl:if test="$optional = 'yes'">  bool <xsl:value-of select="$iport"/>_has_data_;
</xsl:if>
<xsl:text>
  </xsl:text>
  </xsl:for-each>

<xsl:for-each select="/filter/filter-itk/outputs/output">
<xsl:variable name="type-name"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="oport">outport_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="ohandle">outhandle_<xsl:value-of select="@name"/></xsl:variable>
<!-- hard coded datatype --><xsl:text>ITKDatatypeOPort* </xsl:text><xsl:value-of select="$oport"/><xsl:text>_;
  </xsl:text><!-- hard coded datatype --><xsl:text>ITKDatatypeHandle </xsl:text><xsl:value-of select="$ohandle"/><xsl:text>_;

  </xsl:text>
</xsl:for-each>
</xsl:template>



<!-- ====== DEFINE_RUN ====== -->
<xsl:template name="define_run">
<xsl:text disable-output-escaping="yes">
template&lt;</xsl:text>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:choose>
   <xsl:when test="@type"><xsl:value-of select="@type"/><xsl:text> </xsl:text> </xsl:when>
   <xsl:otherwise>class </xsl:otherwise>
</xsl:choose>

<xsl:value-of select="."/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:for-each><xsl:text>&gt;
bool 
</xsl:text><xsl:value-of select="$sci-name"/><xsl:text>::run( </xsl:text>

<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="var">obj_<xsl:value-of select="@name"/> </xsl:variable>itk::Object<xsl:text> *</xsl:text><xsl:value-of select="$var"/>
  <xsl:if test="position() &lt; last()"><xsl:text>, </xsl:text>
  </xsl:if></xsl:for-each>) 
{
<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="data">data_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="obj">obj_<xsl:value-of select="@name"/></xsl:variable>
<xsl:text>  </xsl:text>
  <xsl:value-of select="$type"/> *<xsl:value-of select="$data"/> = dynamic_cast&lt;  <xsl:value-of select="$type"/> * &gt;(<xsl:value-of select="$obj"/>);
  <xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>
  <xsl:choose>
  <xsl:when test="$optional = 'yes'">
  if( inport_<xsl:value-of select="@name"/>_has_data_ ) {
    if( !<xsl:value-of select="$data"/> ) {
      return false;
    }
  }
  </xsl:when>
  <xsl:otherwise>
  if( !<xsl:value-of select="$data"/> ) {
    return false;
  }
</xsl:otherwise>
</xsl:choose>
</xsl:for-each>

  <xsl:if test="$has_defined_objects != ''">
  execute_ = true;
  </xsl:if>
  typedef typename <xsl:value-of select="$itk-name"/>&lt; <xsl:for-each select="/filter/filter-itk/templated/template"><xsl:value-of select="."/>
  <xsl:if test="position() &lt; last()">   
  <xsl:text>, </xsl:text>
  </xsl:if>
 </xsl:for-each> &gt; FilterType;

  // Check if filter_ has been created
  // or the input data has changed. If
  // this is the case, set the inputs.

  if(filter_ == 0<xsl:for-each select="/filter/filter-itk/inputs/input"><xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable> || <xsl:choose>
<xsl:when test="$optional='yes'">
     (inport_<xsl:value-of select="@name"/>_has_data_ &amp;&amp; inhandle_<xsl:value-of select="@name"/>_->generation != last_<xsl:value-of select="@name"/>_)</xsl:when><xsl:otherwise>
     inhandle_<xsl:value-of select="@name"/>_->generation != last_<xsl:value-of select="@name"/>_</xsl:otherwise></xsl:choose></xsl:for-each>) {
     <xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>
<xsl:choose>
<xsl:when test="$optional='yes'">
     if(inport_<xsl:value-of select="@name"/>_has_data_) {
       last_<xsl:value-of select="@name"/>_ = inhandle_<xsl:value-of select="@name"/>_->generation;
     }
</xsl:when>
<xsl:otherwise>
     last_<xsl:value-of select="@name"/>_ = inhandle_<xsl:value-of select="@name"/>_->generation;</xsl:otherwise></xsl:choose>
     </xsl:for-each>

     // create a new one
     filter_ = FilterType::New();

     // attach observer for progress bar
     Observe( filter_.GetPointer() );

     // set inputs 
     <xsl:for-each select="/filter/filter-itk/inputs/input">
  <!-- if input is optional, only call if we have data -->
  <xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>  
  <xsl:choose>
  <xsl:when test="$optional='yes'">
     if( inport_<xsl:value-of select="@name"/>_has_data_ ) {
       dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>( data_<xsl:value-of select="@name"/> );  
     }
  </xsl:when>
  <xsl:otherwise>
     dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>( data_<xsl:value-of select="@name"/> );
  </xsl:otherwise>
  </xsl:choose>
   </xsl:for-each>     
  }

  // reset progress bar
  update_progress(0.0);

  // set filter parameters
   
  <xsl:if test="$has_defined_objects!=''">
  // instantiate any defined objects
  </xsl:if>
  <xsl:for-each select="/filter/filter-itk/parameters/param">
  <xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="name"><xsl:value-of select="name"/></xsl:variable>
  <xsl:variable name="const"><xsl:call-template name="determine_if_const_parameter"/></xsl:variable>
  <xsl:variable name="defined_object">
  <xsl:call-template name="determine_type"/>
  </xsl:variable>
  <xsl:if test="$const != 'yes'">
  <xsl:if test="$defined_object='yes'">
  typename <xsl:value-of select="type"/><xsl:text> </xsl:text><xsl:value-of select="name"/>;
  </xsl:if></xsl:if>  
  </xsl:for-each>

  <xsl:if test="$has_defined_objects != ''">
  // clear defined object guis if things aren't in sync
  <!-- Find the first one and use that in the if statement -->
  <xsl:variable name="control"><xsl:call-template name="create_if"/></xsl:variable>
  <xsl:for-each select="/filter/filter-itk/parameters/param">
  <xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="n"><xsl:value-of select="name"/></xsl:variable>
  <xsl:if test="$control=$n">
  <xsl:variable name="call"><xsl:value-of select="/filter/filter-itk/datatypes/array[@name=$type]/size-call"/></xsl:variable>
  if(<xsl:value-of select="$control"/>.<xsl:value-of select="$call"/>() != gui_dimension_.get()) { 
    gui_dimension_.set(<xsl:value-of select="name"/>.<xsl:value-of select="$call"/>());    
  </xsl:if>
  </xsl:for-each>
  </xsl:if>
  <xsl:if test="$has_defined_objects!=''">
    // for each defined object, clear gui
  </xsl:if>
    <xsl:for-each select="/filter/filter-itk/parameters/param">
  <xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="call"><xsl:value-of select="/filter/filter-itk/datatypes/array[@name=$type]/size-call"/></xsl:variable>
  <xsl:variable name="const"><xsl:call-template name="determine_if_const_parameter"/></xsl:variable>
  <xsl:variable name="defined_object">
  <xsl:call-template name="determine_type"/>
  </xsl:variable>
  <xsl:if test="$const != 'yes'">
  <xsl:if test="$defined_object='yes'">
    gui-&gt;execute(id.c_str() + string(&quot; clear_<xsl:value-of select="name"/>_gui&quot;));
    gui-&gt;execute(id.c_str() + string(&quot; init_<xsl:value-of select="name"/>_dimensions&quot;));
    </xsl:if></xsl:if>
    </xsl:for-each>
  <xsl:if test="$has_defined_objects != ''">
    execute_ = false;
  }
  </xsl:if>
  
  <xsl:for-each select="/filter/filter-itk/parameters/param">
  <xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="name"><xsl:value-of select="name"/></xsl:variable>
  <xsl:variable name="const"><xsl:call-template name="determine_if_const_parameter"/></xsl:variable>
<xsl:variable name="defined_object">
<xsl:call-template name="determine_type"/>
</xsl:variable>

  <xsl:choose>
  <xsl:when test="$defined_object = 'yes'">
  <xsl:variable name="call"><xsl:value-of select="/filter/filter-itk/datatypes/array[@name=$type]/size-call"/></xsl:variable>
  <xsl:variable name="type2"><xsl:value-of select="/filter/filter-itk/datatypes/array[@name=$type]/elem-type"/></xsl:variable>

  <xsl:choose>
  <xsl:when test="$const = 'yes'">
  for(int i=0; i&lt;<xsl:value-of select="name"/>.<xsl:value-of select="$call"/>(); i++) {
    <xsl:value-of select="name"/>[i] = <xsl:value-of select="default"/>;
  }
  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>( <xsl:value-of select="name"/> );

  </xsl:when>
  <xsl:otherwise>
  // register GuiVars
  // avoid pushing onto vector each time if not needed
  int start_<xsl:value-of select="name"/> = 0;
  if(gui_<xsl:value-of select="name"/>_.size() &gt; 0) {
    start_<xsl:value-of select="name"/> = gui_<xsl:value-of select="name"/>_.size();
  }

  for(unsigned int i=start_<xsl:value-of select="name"/>; i&lt;<xsl:value-of select="name"/>.<xsl:value-of select="$call"/>(); i++) {
    ostringstream str;
    str &lt;&lt; &quot;<xsl:value-of select="name"/>&quot; &lt;&lt; i;
<xsl:text>    </xsl:text>
<xsl:choose>
<xsl:when test="$type2='double'">
    gui_<xsl:value-of select="name"/>_.push_back(new GuiDouble(ctx-&gt;subVar(str.str())));
</xsl:when>
<xsl:when test="$type2='float'">
    gui_<xsl:value-of select="name"/>_.push_back(new GuiDouble(ctx-&gt;subVar(str.str())));
</xsl:when>
<xsl:when test="$type2='int'">
    gui_<xsl:value-of select="name"/>_.push_back(new GuiInt(ctx-&gt;subVar(str.str())));
</xsl:when>
<xsl:when test="$type2='bool'">
    gui_<xsl:value-of select="name"/>_.push_back(new GuiInt(ctx-&gt;subVar(str.str())));
</xsl:when>
<xsl:otherwise>
    gui_<xsl:value-of select="name"/>_.push_back(new GuiDouble(ctx-&gt;subVar(str.str())));
</xsl:otherwise>
</xsl:choose>
  }

  // set <xsl:value-of select="name"/> values
  for(int i=0; i&lt;<xsl:value-of select="name"/>.<xsl:value-of select="$call"/>(); i++) {
    <xsl:value-of select="name"/>[i] = gui_<xsl:value-of select="name"/>_[i]-&gt;get();
  }

  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-&gt;<xsl:value-of select="call"/>( <xsl:value-of select="name"/> );

  </xsl:otherwise>
  </xsl:choose>
  </xsl:when>
  <xsl:otherwise>
  <xsl:choose>  

  <xsl:when test="$type='bool'">
  <!-- HANDLE BOOL PARAM (toggle) -->
  <!-- Determine if 1 or 2 call tags -->
  <xsl:choose>
  <xsl:when test="count(call) = 2">
  <xsl:choose>
  <xsl:when test="$const = 'yes'">
  <xsl:variable name="case"><xsl:value-of select="default"/></xsl:variable>
  <xsl:choose>
  <xsl:when test="$case='0'">
  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call[@value='off']"/>( );
  </xsl:when>
  <xsl:otherwise>
  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call[@value='on']"/>( );
  </xsl:otherwise>  
  </xsl:choose>
  </xsl:when>
  <xsl:otherwise>
  if( gui_<xsl:value-of select="name"/>_.get() ) {
    dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call[@value='on']"/>( );   
  } 
  else { 
    dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call[@value='off']"/>( );
  }  
  </xsl:otherwise>
  </xsl:choose>
  </xsl:when>
  <xsl:otherwise>
  <xsl:choose>
  <xsl:when test="$const='yes'">
  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>( <xsl:value-of select="default"/> ); 
  </xsl:when>
  <xsl:otherwise>
  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>( gui_<xsl:value-of select="name"/>_.get() ); 
  </xsl:otherwise>
  </xsl:choose>
  </xsl:otherwise>
  </xsl:choose>

  </xsl:when>
  <xsl:otherwise>
  <xsl:choose>
  <xsl:when test="$const = 'yes'">
  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>( <xsl:value-of select="default"/> ); 
  </xsl:when>
  <xsl:otherwise>
  dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>( gui_<xsl:value-of select="name"/>_.get() ); 
  </xsl:otherwise>
  </xsl:choose>
  </xsl:otherwise>
  </xsl:choose>
  </xsl:otherwise>
  </xsl:choose>
  </xsl:for-each>

  // execute the filter
  <xsl:if test="$has_defined_objects != ''">
  if (execute_) {
  </xsl:if>
  try {

    dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())->Update();

  } catch ( itk::ExceptionObject &amp; err ) {
     error("ExceptionObject caught!");
     error(err.GetDescription());
  }

  // get filter output
  <xsl:for-each select="/filter/filter-itk/outputs/output">
<xsl:variable name="ohandle">outhandle_<xsl:value-of select="@name"/>_</xsl:variable>
<xsl:variable name="oport">outport_<xsl:value-of select="@name"/>_</xsl:variable>
  <xsl:variable name="const"><xsl:value-of select="call/@const"/></xsl:variable>
  <xsl:variable name="name"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="output"><!-- hard coded datatype -->ITKDatatype</xsl:variable>
  <!-- Declare ITKDatatye hard coded datatype -->
  ITKDatatype* out_<xsl:value-of select="@name"/>_ = scinew ITKDatatype; 
  <xsl:choose>
  <xsl:when test="$const = 'yes'">
  out_<xsl:value-of select="@name"/>_->data_ = const_cast&lt;<xsl:value-of select="type"/>*  &gt;(dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>());
  </xsl:when>
  <xsl:otherwise>
  out_<xsl:value-of select="@name"/>_->data_ = dynamic_cast&lt;FilterType* &gt;(filter_.GetPointer())-><xsl:value-of select="call"/>();
  </xsl:otherwise>
  </xsl:choose>
  outhandle_<xsl:value-of select="@name"/>_ = out_<xsl:value-of select="@name"/>_; 
  <xsl:value-of select="$oport"/><xsl:text>->send(</xsl:text><xsl:value-of select="$ohandle"/>
<xsl:text>);
  </xsl:text>
  </xsl:for-each>
  <xsl:if test="$has_defined_objects!=''">
  }
  </xsl:if>

  return true;
<xsl:text>}
</xsl:text>
</xsl:template>



<!-- ====== DEFINE_CONSTRUCTOR ====== -->
<xsl:template name="define_constructor">
<xsl:value-of select="$sci-name"/>
<xsl:text>::</xsl:text><xsl:value-of select="$sci-name"/>
<xsl:text disable-output-escaping="yes">(GuiContext* ctx)
  : Module(&quot;</xsl:text>
<xsl:value-of select="$sci-name"/>
<xsl:text disable-output-escaping="yes">&quot;, ctx, Source, &quot;</xsl:text>
<xsl:value-of select="$category"/><xsl:text>&quot;, &quot;</xsl:text>
<xsl:value-of select="$package"/><xsl:text>&quot;)</xsl:text>
<xsl:for-each select="/filter/filter-itk/parameters/param">
<xsl:variable name="const"><xsl:call-template name="determine_if_const_parameter"/></xsl:variable>
<xsl:variable name="defined_object">
<xsl:call-template name="determine_type"/>
</xsl:variable>
<xsl:if test="$const!='yes'">
<xsl:if test="$defined_object = 'no'">
<xsl:text>,
     gui_</xsl:text><xsl:value-of select="name"/>
<xsl:text disable-output-escaping="yes">_(ctx->subVar(&quot;</xsl:text>
<xsl:value-of select="name"/>
<xsl:text disable-output-escaping="yes">&quot;))</xsl:text></xsl:if>
</xsl:if>
</xsl:for-each><xsl:for-each select="/filter/filter-sci/outputs/output"><xsl:variable name="send"><xsl:value-of select="@send_intermediate"/></xsl:variable><xsl:if test="$send='yes'"><xsl:text>,
     gui_</xsl:text>update_<xsl:value-of select="@name"/>_(ctx->subVar(&quot;update_<xsl:value-of select="@name"/>&quot;))<xsl:text>,
     gui_</xsl:text>update_iters_<xsl:value-of select="@name"/>_(ctx->subVar(&quot;update_iters_<xsl:value-of select="@name"/>&quot;))<!-- FIX ME --></xsl:if></xsl:for-each>

<xsl:if test="$has_defined_objects != ''">,
     gui_dimension_(ctx-&gt;subVar(&quot;dimension&quot;))</xsl:if>
<xsl:for-each select="/filter/filter-itk/inputs/input">, 
     last_<xsl:value-of select="@name"/>_(-1)</xsl:for-each>
<xsl:text>
{
  filter_ = 0;
</xsl:text>
<xsl:for-each select="/filter/filter-itk/inputs/input">
  <xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>
  <xsl:choose>
  <xsl:when test="$optional = 'yes'">  inport_<xsl:value-of select="@name"/>_has_data_ = false;  </xsl:when>
  <xsl:otherwise>
</xsl:otherwise>
</xsl:choose></xsl:for-each>
<xsl:if test="$has_defined_objects != ''">
  gui_dimension_.set(0);</xsl:if>
<xsl:text>
</xsl:text>
  m_RedrawCommand = RedrawCommandType::New();
  m_RedrawCommand-&gt;SetCallbackFunction( this, &amp;<xsl:value-of select="/filter/filter-sci/@name"/>::ProcessEvent );
  m_RedrawCommand-&gt;SetCallbackFunction( this, &amp;<xsl:value-of select="/filter/filter-sci/@name"/>::ConstProcessEvent );
<xsl:for-each select="/filter/filter-sci/outputs/output">
<xsl:variable name="send"><xsl:value-of select="@send_intermediate"/></xsl:variable>
<xsl:if test="$send='yes'">
  iterationCounter_<xsl:value-of select="@name"/> = 0;
</xsl:if>
</xsl:for-each>
  update_progress(0.0);
<xsl:text>
}

</xsl:text>
</xsl:template>



<!-- ====== DEFINE_DESTRUCTOR ====== -->
<xsl:template name="define_destructor">
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text>::~</xsl:text><xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text>() 
{
}

</xsl:text>
</xsl:template>



<!-- ====== DEFINE_EXECUTE ====== -->
<xsl:template name="define_execute">
<xsl:text>void 
</xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text>::execute() 
{
  // check input ports
</xsl:text>
<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="type-name"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="iport">inport_<xsl:value-of select="@name"/>_</xsl:variable>
<xsl:variable name="ihandle">inhandle_<xsl:value-of select="@name"/>_</xsl:variable>
<xsl:variable name="port-type"><!-- hard coded datatype -->ITKDatatype<xsl:text>IPort</xsl:text></xsl:variable>
<xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>

<xsl:text>  </xsl:text><xsl:value-of select="$iport"/><xsl:text> = (</xsl:text>
<xsl:value-of select="$port-type"/><xsl:text> *)get_iport(&quot;</xsl:text><xsl:value-of select="@name"/><xsl:text>&quot;);
  if(!</xsl:text><xsl:value-of select="$iport"/><xsl:text>) {
    error(&quot;Unable to initialize iport&quot;);
    return;
  }

  </xsl:text>
<xsl:value-of select="$iport"/><xsl:text>->get(</xsl:text>
<xsl:value-of select="$ihandle"/><xsl:text>);
</xsl:text>
<xsl:choose>
  <xsl:when test="$optional = 'yes'">
  if(!<xsl:value-of select="$ihandle"/><xsl:text>.get_rep()) {
    remark("No data in optional </xsl:text><xsl:value-of select="$iport"/>!");			       
    <xsl:value-of select="$iport"/>has_data_ = false;
  }
  else {
    <xsl:value-of select="$iport"/>has_data_ = true;
  }

  </xsl:when>
  <xsl:otherwise>
  if(!<xsl:value-of select="$ihandle"/><xsl:text>.get_rep()) {
    return;
  }

</xsl:text>
</xsl:otherwise>
</xsl:choose>
</xsl:for-each>

<xsl:text>
  // check output ports
</xsl:text>
<xsl:for-each select="/filter/filter-itk/outputs/output">
<xsl:variable name="type-name"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="oport">outport_<xsl:value-of select="@name"/>_</xsl:variable>
<xsl:variable name="ohandle">outhandle_<xsl:value-of select="@name"/>_</xsl:variable>
<xsl:variable name="port-type"><!-- hard coded datatype -->ITKDatatype<xsl:text>OPort</xsl:text></xsl:variable>

<xsl:text>  </xsl:text><xsl:value-of select="$oport"/><xsl:text> = (</xsl:text>
<xsl:value-of select="$port-type"/><xsl:text> *)get_oport(&quot;</xsl:text><xsl:value-of select="@name"/><xsl:text>&quot;);
  if(!</xsl:text><xsl:value-of select="$oport"/><xsl:text>) {
    error(&quot;Unable to initialize oport&quot;);
    return;
  }
</xsl:text>
</xsl:for-each>

<xsl:for-each select="/filter/filter-sci/outputs/output">
<xsl:variable name="send"><xsl:value-of select="@send_intermediate"/></xsl:variable>
<xsl:if test="$send='yes'">
  iterationCounter_<xsl:value-of select="@name"/> = 0;	
  gui_update_<xsl:value-of select="@name"/>_.reset();
  gui_update_iters_<xsl:value-of select="@name"/>_.reset();
</xsl:if>
</xsl:for-each>
<xsl:text>
  // get input
  </xsl:text>
<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="ihandle">inhandle_<xsl:value-of select="@name"/>_</xsl:variable>
<xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>
<xsl:choose>
  <xsl:when test="$optional = 'yes'">
  itk::Object<xsl:text>* data_</xsl:text><xsl:value-of select="@name"/> = 0;
  if( inport_<xsl:value-of select="@name"/>_has_data_ ) {
    data_<xsl:value-of select="@name"/> = <xsl:value-of select="$ihandle"/>.get_rep()->data_.GetPointer();
  }
  </xsl:when>
  <xsl:otherwise>itk::Object<xsl:text>* data_</xsl:text><xsl:value-of select="@name"/><xsl:text> = </xsl:text><xsl:value-of select="$ihandle"/><xsl:text>.get_rep()->data_.GetPointer();
  </xsl:text>
  </xsl:otherwise>
</xsl:choose>
</xsl:for-each>
<xsl:variable name="defaults"><xsl:value-of select="/filter/filter-sci/instantiations/@use-defaults"/></xsl:variable>
<xsl:text>
  // can we operate on it?
  if(0) { }</xsl:text>
  <xsl:choose>
    <xsl:when test="$defaults = 'yes'">
<xsl:for-each select="/filter/filter-itk/templated/defaults"><xsl:text>
  else if(run&lt; </xsl:text>
  <xsl:for-each select="default">
  <xsl:value-of select="."/><xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
    </xsl:for-each>
<xsl:text> &gt;( </xsl:text>
<xsl:for-each select="/filter/filter-itk/inputs/input">data_<xsl:value-of select="@name"/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if></xsl:for-each>
    <xsl:text> )) {} </xsl:text>
</xsl:for-each>
    </xsl:when>
    <xsl:otherwise>
  <xsl:for-each select="/filter/filter-sci/instantiations/instance">
  <xsl:variable name="num"><xsl:value-of select="position()"/></xsl:variable>
  <xsl:text> 
  else if(run&lt; </xsl:text>
<xsl:for-each select="type">
<xsl:variable name="type"><xsl:value-of select="@name"/></xsl:variable>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:variable name="templated_type"><xsl:value-of select="."/></xsl:variable>	      
<xsl:if test="$type = $templated_type">
<xsl:value-of select="/filter/filter-sci/instantiations/instance[position()=$num]/type[@name=$type]/value"/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:if>
</xsl:for-each>
</xsl:for-each>
<xsl:text> &gt;( </xsl:text>
<xsl:for-each select="/filter/filter-itk/inputs/input">data_<xsl:value-of select="@name"/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if></xsl:for-each><xsl:text> )) { }</xsl:text></xsl:for-each>
</xsl:otherwise>
</xsl:choose>
<xsl:text>
  else {
    // error
    error(&quot;Incorrect input type&quot;);
    return;
  }
</xsl:text>
<xsl:text>
}

</xsl:text>
</xsl:template>



<!-- ======= DEFINE_PROCESSEVENT ======= -->
<xsl:template name="define_processevent">
// Manage a Progress event 
void 
<xsl:value-of select="/filter/filter-sci/@name"/>::ProcessEvent( itk::Object * caller, const itk::EventObject &amp; event )
{
  if( typeid( itk::ProgressEvent )   ==  typeid( event ) )
  {
    ::itk::ProcessObject::Pointer  process = 
        dynamic_cast&lt; itk::ProcessObject *&gt;( caller );

    const double value = static_cast&lt;double&gt;(process->GetProgress() );
    update_progress( value );
    }
<xsl:if test="/filter/filter-sci/outputs/output">
  else if ( typeid( itk::IterationEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::Pointer  process = 
	dynamic_cast&lt; itk::ProcessObject *&gt;( caller );
    
    update_after_iteration();
  }
</xsl:if>
}

</xsl:template>

<!-- ======= DEFINE_CONSTPROCESSEVENT ======= -->
<xsl:template name="define_constprocessevent">
// Manage a Progress event 
void 
<xsl:value-of select="/filter/filter-sci/@name"/>::ConstProcessEvent(const itk::Object * caller, const itk::EventObject &amp; event )
{
  if( typeid( itk::ProgressEvent )   ==  typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer  process = 
        dynamic_cast&lt; const itk::ProcessObject *&gt;( caller );

    const double value = static_cast&lt;double&gt;(process->GetProgress() );
    update_progress( value );
    }
<xsl:if test="/filter/filter-sci/outputs/output">
  else if ( typeid( itk::IterationEvent ) == typeid( event ) )
  {
    ::itk::ProcessObject::ConstPointer  process = 
	dynamic_cast&lt; const itk::ProcessObject *&gt;( caller );
    
    update_after_iteration();
  }
</xsl:if>
}

</xsl:template>


<!-- ======= DEFINE_UPDATE_AFTER_ITERATION ======= -->
<xsl:template name="define_update_after_iteration">
void 
<xsl:value-of select="/filter/filter-sci/@name"/>::update_after_iteration()
{
<xsl:for-each select="/filter/filter-sci/outputs/output">
<xsl:variable name="send"><xsl:value-of select="@send_intermediate"/></xsl:variable>
<xsl:variable name="funcName"><xsl:value-of select="@name"/></xsl:variable>
<xsl:if test="$send='yes'">
  if(gui_update_<xsl:value-of select="$funcName"/>_.get() &amp;&amp; iterationCounter_<xsl:value-of select="$funcName"/>%gui_update_iters_<xsl:value-of select="$funcName"/>_.get() == 0 &amp;&amp; iterationCounter_<xsl:value-of select="$funcName"/> &gt; 0) {
<xsl:variable name="defaults"><xsl:value-of select="/filter/filter-sci/instantiations/@use-defaults"/></xsl:variable>
    // determine type and call do it
    if(0) { } 
  <xsl:choose>
    <xsl:when test="$defaults = 'yes'">
<xsl:for-each select="/filter/filter-itk/templated/defaults"><xsl:text>
    else if(do_it_</xsl:text><xsl:value-of select="$funcName"/><xsl:text>&lt; </xsl:text>
  <xsl:for-each select="default">
  <xsl:value-of select="."/><xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
    </xsl:for-each>
<xsl:text> &gt;( )) {} </xsl:text>
</xsl:for-each>
    </xsl:when>
    <xsl:otherwise>
  <xsl:for-each select="/filter/filter-sci/instantiations/instance">
  <xsl:variable name="num"><xsl:value-of select="position()"/></xsl:variable>
  <xsl:text> 
    else if(do_it_</xsl:text><xsl:value-of select="$funcName"/><xsl:text>&lt; </xsl:text>
<xsl:for-each select="type">
<xsl:variable name="type"><xsl:value-of select="@name"/></xsl:variable>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:variable name="templated_type"><xsl:value-of select="."/></xsl:variable>	      
<xsl:if test="$type = $templated_type">
<xsl:value-of select="/filter/filter-sci/instantiations/instance[position()=$num]/type[@name=$type]/value"/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:if>
</xsl:for-each>
</xsl:for-each>
<xsl:text> &gt;( )) { }</xsl:text></xsl:for-each>
</xsl:otherwise>
</xsl:choose>
    else {
      // error
      error("Incorrect filter type");
      return;
    }
  }
  iterationCounter_<xsl:value-of select="@name"/>++;
</xsl:if>
</xsl:for-each>
}

</xsl:template>



<!-- ======= DEFINE_DO_ITS ======= -->
<xsl:template name="define_do_its">
<xsl:for-each select="/filter/filter-sci/outputs/output">
<xsl:variable name="send"><xsl:value-of select="@send_intermediate"/></xsl:variable>
<xsl:if test="$send='yes'"><xsl:text>
template&lt;</xsl:text>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:choose>
   <xsl:when test="@type"><xsl:value-of select="@type"/><xsl:text> </xsl:text> </xsl:when>
   <xsl:otherwise>class </xsl:otherwise>
</xsl:choose> 
<xsl:value-of select="."/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:for-each>&gt;
bool 
<xsl:value-of select="/filter/filter-sci/@name"/>::do_it_<xsl:value-of select="@name"/>()
{
  // Move the pixel container and image information of the image 
  // we are working on into a temporary image to  use as the 
  // input to the mini-pipeline.  This avoids a complete copy of the image.

  typedef typename <xsl:value-of select="$itk-name"/>&lt; <xsl:for-each select="/filter/filter-itk/templated/template"><xsl:value-of select="."/>
  <xsl:if test="position() &lt; last()">   
  <xsl:text>, </xsl:text>
  </xsl:if>
 </xsl:for-each> &gt; FilterType;
  
  if(!dynamic_cast&lt;FilterType*&gt;(filter_.GetPointer())) {
    return false;
  }
 
  <xsl:variable name="name"><xsl:value-of select="@name"/></xsl:variable> 
  <xsl:variable name="type"><xsl:value-of select="/filter/filter-itk/outputs/output[@name=$name]/type"/></xsl:variable>
  typename <xsl:value-of select="$type"/>::Pointer tmp = <xsl:value-of select="$type"/>::New();
  tmp->SetRequestedRegion( dynamic_cast&lt;FilterType*&gt;(filter_.GetPointer())->GetOutput()->GetRequestedRegion() );
  tmp->SetBufferedRegion( dynamic_cast&lt;FilterType*&gt;(filter_.GetPointer())->GetOutput()->GetBufferedRegion() );
  tmp->SetLargestPossibleRegion( dynamic_cast&lt;FilterType*&gt;(filter_.GetPointer())->GetOutput()->GetLargestPossibleRegion() );
  tmp->SetPixelContainer( dynamic_cast&lt;FilterType*&gt;(filter_.GetPointer())->GetOutput()->GetPixelContainer() );
  tmp->CopyInformation( dynamic_cast&lt;FilterType*&gt;(filter_.GetPointer())->GetOutput() );
  
  
  // send segmentation down
  ITKDatatype* out_<xsl:value-of select="@name"/>_ = scinew ITKDatatype; 
  out_OutputImage_->data_ = tmp;
  outhandle_<xsl:value-of select="@name"/>_ = out_<xsl:value-of select="@name"/>_; 
  outport_<xsl:value-of select="@name"/>_->send_intermediate(outhandle_<xsl:value-of select="@name"/>_);
  return true;
}
</xsl:if>

</xsl:for-each>

</xsl:template>




<!-- ======= DEFINE_OBSERVE ======= -->
<xsl:template name="define_observe">
// Manage a Progress event 
void 
<xsl:value-of select="/filter/filter-sci/@name"/>::Observe( itk::Object *caller )
{
  caller->AddObserver(  itk::ProgressEvent(), m_RedrawCommand.GetPointer() );
  caller->AddObserver(  itk::IterationEvent(), m_RedrawCommand.GetPointer() );
}

</xsl:template>




<!-- ====== DEFINE_TCL_COMMAND ====== -->
<xsl:template name="define_tcl_command">
<xsl:text>void 
</xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text disable-output-escaping="yes">::tcl_command(GuiArgs&amp; args, void* userdata)
{
  Module::tcl_command(args, userdata);
</xsl:text>

<xsl:text>
}

</xsl:text>
</xsl:template>



<!-- Helper function to determine if a parameter is a primitive type or defined type -->
<xsl:template name="determine_type">
<xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
<xsl:choose>
<xsl:when test="$type='int'">no</xsl:when>
<xsl:when test="$type='float'">no</xsl:when>
<xsl:when test="$type='double'">no</xsl:when>
<xsl:when test="$type='bool'">no</xsl:when>
<xsl:when test="$type='char'">no</xsl:when>
<xsl:when test="$type='unsigned char'">no</xsl:when>
<xsl:when test="$type='short'">no</xsl:when>
<xsl:when test="$type='unsigned short'">no</xsl:when>
<xsl:otherwise>yes</xsl:otherwise>
</xsl:choose>
</xsl:template>



<!-- Helper function to determine if a parameter has been defined as const in the gui filter xml file.  If it has, a specified value will always be set and no gui will be visible to the user.
-->
<xsl:template name="determine_if_const_parameter">
<xsl:variable name="name"><xsl:value-of select="name"/></xsl:variable>
<xsl:variable name="const"><xsl:value-of select="/filter/filter-gui/param[@name=$name]/const/@value"/></xsl:variable>
<xsl:choose>
<xsl:when test="$const != ''">yes</xsl:when>
<xsl:otherwise>no</xsl:otherwise>
</xsl:choose>

</xsl:template>


<!-- This doesn't need a "break;" -->
<xsl:template name="create_if">
  <xsl:apply-templates select="/filter/filter-itk/parameters/param[1]" mode="create_if" />
</xsl:template>

<xsl:template match="/filter/filter-itk/parameters/param" mode="create_if">
  <xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="defined_object">
  <xsl:call-template name="determine_type"/>
  </xsl:variable>
  <xsl:choose>
  <xsl:when test="$defined_object='yes'">
    <xsl:value-of select="name"/>
  </xsl:when>
  <xsl:otherwise>
  <xsl:variable
    name="next"
    select="following-sibling::param[position() &gt; 1]" />
    <xsl:if test="$next">
      <xsl:apply-templates select="$next" mode="create_if" />
    </xsl:if>
  </xsl:otherwise>
  </xsl:choose>
</xsl:template>

</xsl:stylesheet>


