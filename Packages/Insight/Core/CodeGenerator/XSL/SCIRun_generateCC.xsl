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
<xsl:for-each select="/filter/filter-sci/includes/file">
#include &lt;<xsl:value-of select="."/>&gt;
</xsl:for-each>
<xsl:for-each select="/filter/filter-itk/includes/file">
#include &lt;<xsl:value-of select="."/>&gt;
</xsl:for-each>
</xsl:template>



<!-- ====== CREATE_CLASS_DECL ====== -->
<xsl:template name="create_class_decl">
<xsl:text>class </xsl:text><xsl:value-of select="$package"/><xsl:text>SHARE </xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text> : public Module 
{
public:

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
<xsl:variable name="type"><xsl:value-of select="@type"/></xsl:variable>
<xsl:choose>
   <xsl:when test="$type = ''">class </xsl:when>
   <xsl:otherwise><xsl:value-of select="$type"/><xsl:text> </xsl:text> </xsl:otherwise>
</xsl:choose> 
<xsl:value-of select="."/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:for-each><xsl:text> &gt; 
  bool run( </xsl:text>
<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="import"><xsl:value-of select="@import"/></xsl:variable>
<xsl:choose>
  <xsl:when test="$import = 'yes'">
  <xsl:variable name="import_name"><xsl:value-of select="type"/></xsl:variable>itk::Object*</xsl:when>
  <xsl:otherwise>itk::Object* </xsl:otherwise>
</xsl:choose>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if></xsl:for-each> );
};

</xsl:template>



<!-- ====== DECLARE_GUIVARS ====== -->
<xsl:template name="declare_guivars">
<xsl:for-each select="/filter[@name=$root-name]/filter-itk[@name=$itk-name]/parameters/param">
<xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="defined_object"><xsl:value-of select="@defined"/></xsl:variable>
<xsl:choose>
<xsl:when test="$defined_object = 'yes'">
<xsl:variable name="data_type"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="type2"><xsl:value-of select="/filter/filter-itk/datatypes/datatype[@name=$data_type]/field/type"/></xsl:variable>

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
<xsl:otherwise>
GuiDouble </xsl:otherwise>
</xsl:choose>
</xsl:otherwise>
</xsl:choose>
<xsl:text> gui_</xsl:text><xsl:value-of select="name"/>
<xsl:text>_;
  </xsl:text>
</xsl:for-each>
<xsl:if test="$has_defined_objects != ''">
  GuiInt gui_dimension_;</xsl:if>
</xsl:template>




<!-- ====== DECLARE_PORTS_AND_HANDLES ====== -->
<xsl:template name="declare_ports_and_handles">

<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="type-name"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="iport">inport_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="ihandle">inhandle_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="import"><xsl:value-of select="@import"/></xsl:variable>
<xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>
<xsl:choose>
  <xsl:when test="$import = 'yes'">
  <xsl:variable name="import_filter"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="import_port"><xsl:value-of select="port"/></xsl:variable>
  <xsl:variable name="import_type"><xsl:value-of select="/filter/filter-itk/filter[@name=$import_filter]/filter-itk/inputs/input[@name=$import_port]/type"/></xsl:variable><!-- hard coded datatype --><xsl:text>  </xsl:text>ITKDatatype<xsl:text>IPort* </xsl:text><xsl:value-of select="$iport"/><xsl:text>_;
  </xsl:text><!-- hard coded datatype --><xsl:text>ITKDatatypeHandle </xsl:text><xsl:value-of select="$ihandle"/>_;
  </xsl:when>
  <xsl:otherwise>
<!-- hard coded datatype --><xsl:text>ITKDatatypeIPort* </xsl:text><xsl:value-of select="$iport"/><xsl:text>_;
  </xsl:text><!-- hard coded datatype --><xsl:text>ITKDatatypeHandle </xsl:text><xsl:value-of select="$ihandle"/>_;
  </xsl:otherwise>
</xsl:choose>

<xsl:if test="$optional = 'yes'">  bool <xsl:value-of select="$iport"/>_has_data_;
</xsl:if>
<xsl:text>
  </xsl:text>
</xsl:for-each>

<xsl:for-each select="/filter/filter-itk/outputs/output">
<xsl:variable name="type-name"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="oport">outport_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="ohandle">outhandle_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="import"><xsl:value-of select="@import"/></xsl:variable>
<xsl:choose>
  <xsl:when test="$import = 'yes'">
  <xsl:variable name="import_filter"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="import_port"><xsl:value-of select="port"/></xsl:variable>
  <xsl:variable name="import_type"><xsl:value-of select="/filter/filter-itk/filter[@name=$import_filter]/filter-itk/outputs/output[@name=$import_port]/type"/></xsl:variable>
<!-- hard coded datatype --><xsl:text>ITKDatatypeIPort* </xsl:text><xsl:value-of select="$oport"/><xsl:text>_;
  </xsl:text><!-- hard coded datatype --><xsl:text>ITKDatatypeHandle </xsl:text><xsl:value-of select="$ohandle"/>_;
  </xsl:when>
  <xsl:otherwise>
<!-- hard coded datatype --><xsl:text>ITKDatatypeOPort* </xsl:text><xsl:value-of select="$oport"/><xsl:text>_;
  </xsl:text><!-- hard coded datatype --><xsl:text>ITKDatatypeHandle </xsl:text><xsl:value-of select="$ohandle"/><xsl:text>_;

  </xsl:text>
  </xsl:otherwise>
</xsl:choose></xsl:for-each>
</xsl:template>



<!-- ====== DEFINE_RUN ====== -->
<xsl:template name="define_run">
<xsl:text disable-output-escaping="yes">
template&lt;</xsl:text>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:variable name="type"><xsl:value-of select="@type"/></xsl:variable>
<xsl:choose>
   <xsl:when test="$type = ''">class </xsl:when>
   <xsl:otherwise><xsl:value-of select="$type"/><xsl:text> </xsl:text> </xsl:otherwise>
</xsl:choose>

<xsl:value-of select="."/>
<xsl:if test="position() &lt; last()">
<xsl:text>, </xsl:text>
</xsl:if>
</xsl:for-each><xsl:text>&gt;
bool </xsl:text><xsl:value-of select="$sci-name"/><xsl:text>::run( </xsl:text>

<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="var">obj_<xsl:value-of select="@name"/> </xsl:variable>
<xsl:variable name="import"><xsl:value-of select="@import"/></xsl:variable>
<xsl:choose>
  <xsl:when test="$import = 'yes'">
  <xsl:variable name="import_name"><xsl:value-of select="type"/></xsl:variable>itk::Object* </xsl:when>
  <xsl:otherwise>itk::Object<xsl:text> *</xsl:text>
  </xsl:otherwise>
</xsl:choose>
<xsl:value-of select="$var"/>
  <xsl:if test="position() &lt; last()"><xsl:text>, </xsl:text>
  </xsl:if>
</xsl:for-each>) 
{
<xsl:for-each select="/filter/filter-itk/inputs/input">
<xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
<xsl:variable name="data">data_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="obj">obj_<xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="import"><xsl:value-of select="@import"/></xsl:variable>
<xsl:text>  </xsl:text>
<xsl:choose>
  <xsl:when test="$import = 'yes'">
  <xsl:variable name="import_filter"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="port"><xsl:value-of select="port"/></xsl:variable>
  <xsl:value-of select="/filter/filter-itk/filter[@name=$import_filter]/filter-itk/inputs/input[@name=$port]/type"/>
  </xsl:when>
  <xsl:otherwise>
  <xsl:value-of select="$type"/> *<xsl:value-of select="$data"/> = dynamic_cast&lt;  <xsl:value-of select="$type"/> * &gt;(<xsl:value-of select="$obj"/>);
</xsl:otherwise>
</xsl:choose>
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

  typedef typename <xsl:value-of select="$itk-name"/>&lt; <xsl:for-each select="/filter/filter-itk/templated/template"><xsl:value-of select="."/>
  <xsl:if test="position() &lt; last()">   
  <xsl:text>, </xsl:text>
  </xsl:if>
 </xsl:for-each> &gt; FilterType;

  // create a new filter
  typename FilterType::Pointer filter = FilterType::New();

  // set filter 
 <xsl:if test="$has_defined_objects != ''">
         int dim = 0;</xsl:if>
  <xsl:for-each select="/filter/filter-itk/parameters/param">
  <xsl:variable name="type"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="name"><xsl:value-of select="name"/></xsl:variable>
<xsl:variable name="defined_object"><xsl:value-of select="@defined"/></xsl:variable>

  <xsl:choose>
  <xsl:when test="$defined_object = 'yes'">
  <xsl:variable name="call"><xsl:value-of select="/filter/filter-itk/datatypes/datatype[@name=$type]/field/variable-call"/></xsl:variable>
  <xsl:variable name="type2"><xsl:value-of select="/filter/filter-itk/datatypes/datatype[@name=$type]/field/type"/></xsl:variable>

  <xsl:value-of select="type"/><xsl:text> </xsl:text><xsl:value-of select="name"/>;
  if(<xsl:value-of select="name"/>.<xsl:value-of select="$call"/>() != gui_dimension_.get()) {
	gui-&gt;execute(id.c_str() + string(&quot; clear_gui&quot;));
        gui_dimension_.set(<xsl:value-of select="name"/>.<xsl:value-of select="$call"/>());
	gui-&gt;execute(id.c_str() + string(&quot; init_<xsl:value-of select="name"/>_dimensions&quot;));
  }
  
  // register GuiVars
  // avoid pushing onto vector each time if not needed
  int start = 0;
  if(gui_<xsl:value-of select="name"/>_.size() &gt; 0) {
    start = gui_<xsl:value-of select="name"/>_.size();
  }

  for(int i=start; i&lt;<xsl:value-of select="name"/>.<xsl:value-of select="$call"/>(); i++) {
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

  filter-&gt;<xsl:value-of select="call"/>( <xsl:value-of select="name"/> );

  </xsl:when>
  <xsl:otherwise>
  <xsl:choose>
  <xsl:when test="$type='bool'">
  <!-- HANDLE BOOL PARAM (toggle) -->
  <!-- Determine if 1 or 2 call tags -->
  <xsl:choose>
  <xsl:when test="count(call) = 2">
  if( gui_<xsl:value-of select="name"/>_-&gt;get() ) {
    filter-><xsl:value-of select="call[@value='on']"/>( );   
  } 
  else { 
    filter-><xsl:value-of select="call[@value='off']"/>( );
  }  
  </xsl:when>
  <xsl:otherwise>
  filter-><xsl:value-of select="call"/>( gui_<xsl:value-of select="name"/>_.get() ); 
  </xsl:otherwise>
  </xsl:choose>

  </xsl:when>
  <xsl:otherwise>
  filter-><xsl:value-of select="call"/>( gui_<xsl:value-of select="name"/>_.get() ); 
  </xsl:otherwise>
  </xsl:choose>
  </xsl:otherwise>
  </xsl:choose>
  </xsl:for-each>
<xsl:text>   
  // set inputs 
</xsl:text>
  <xsl:for-each select="/filter/filter-itk/inputs/input">
  <!-- if input is optional, only call if we have data -->
  <xsl:variable name="optional"><xsl:value-of select="@optional"/></xsl:variable>  
  <xsl:choose>
  <xsl:when test="$optional='yes'">
  if( inport_<xsl:value-of select="@name"/>_has_data_ ) {
    filter-><xsl:value-of select="call"/>( data_<xsl:value-of select="@name"/> );  
  }
  </xsl:when>
  <xsl:otherwise>
  filter-><xsl:value-of select="call"/>( data_<xsl:value-of select="@name"/> );
  </xsl:otherwise>
  </xsl:choose>
   </xsl:for-each>

  // execute the filter
  try {

    filter->Update();

  } catch ( itk::ExceptionObject &amp; err ) {
     error("ExceptionObject caught!");
     error(err.GetDescription());
  }

  // get filter output
  <xsl:for-each select="/filter/filter-itk/outputs/output">
  <xsl:variable name="const"><xsl:value-of select="call/@const"/></xsl:variable>
  <xsl:variable name="name"><xsl:value-of select="type"/></xsl:variable>
  <xsl:variable name="output"><!-- hard coded datatype -->ITKDatatype</xsl:variable>
  <!-- Declare ITKDatatye hard coded datatype -->
  ITKDatatype* out_<xsl:value-of select="@name"/>_ = scinew ITKDatatype; 
  <xsl:choose>
  <xsl:when test="$const = 'yes'">
  out_<xsl:value-of select="@name"/>_->data_ = const_cast&lt;<xsl:value-of select="type"/>*  &gt;(filter-><xsl:value-of select="call"/>());
  </xsl:when>
  <xsl:otherwise>
  out_<xsl:value-of select="@name"/>_->data_ = filter-><xsl:value-of select="call"/>();
  </xsl:otherwise>
  </xsl:choose>
  outhandle_<xsl:value-of select="@name"/>_ = out_<xsl:value-of select="@name"/>_; 
  </xsl:for-each>
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
<xsl:variable name="defined_object"><xsl:value-of select="@defined"/></xsl:variable>
<xsl:if test="$defined_object = ''">
<xsl:text>,
     gui_</xsl:text><xsl:value-of select="name"/>
<xsl:text disable-output-escaping="yes">_(ctx->subVar(&quot;</xsl:text>
<xsl:value-of select="name"/>
<xsl:text disable-output-escaping="yes">&quot;))</xsl:text></xsl:if>
</xsl:for-each>
<xsl:if test="$has_defined_objects != ''">,
    gui_dimension_(ctx-&gt;subVar(&quot;dimension&quot;))</xsl:if>
<xsl:text>
{
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
<xsl:text>void </xsl:text>
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
    error("No data in </xsl:text><xsl:value-of select="$iport"/><xsl:text>!");			       
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
  <xsl:variable name="num"><xsl:value-of select="@num"/></xsl:variable>
  <xsl:text> 
  else if(run&lt; </xsl:text>
<xsl:for-each select="type">
<xsl:variable name="type"><xsl:value-of select="@name"/></xsl:variable>
<xsl:for-each select="/filter/filter-itk/templated/template">
<xsl:variable name="templated_type"><xsl:value-of select="."/></xsl:variable>	      
<xsl:if test="$type = $templated_type">
<xsl:value-of select="/filter/filter-sci/instantiations/instance[@num=$num]/type[@name=$type]/value"/>
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

  // send the data downstream
  </xsl:text>
<xsl:for-each select="/filter/filter-itk/outputs/output">
<xsl:variable name="ohandle">outhandle_<xsl:value-of select="@name"/>_</xsl:variable>
<xsl:variable name="oport">outport_<xsl:value-of select="@name"/>_</xsl:variable>

<xsl:value-of select="$oport"/><xsl:text>->send(</xsl:text><xsl:value-of select="$ohandle"/>
<xsl:text>);
  </xsl:text>
</xsl:for-each>
<xsl:text>
}

</xsl:text>
</xsl:template>



<!-- ====== DEFINE_TCL_COMMAND ====== -->
<xsl:template name="define_tcl_command">
<xsl:text>void </xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text disable-output-escaping="yes">::tcl_command(GuiArgs&amp; args, void* userdata)
{
  Module::tcl_command(args, userdata);
</xsl:text>

<xsl:text>
}

</xsl:text>
</xsl:template>


</xsl:stylesheet>
