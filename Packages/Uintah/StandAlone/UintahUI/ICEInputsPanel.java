/**************************************************************************
// Program : ICEInputsPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           ICE flags and parameters
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.event.*;
import java.text.DecimalFormat;
import java.util.Vector;

//**************************************************************************
// Class   : ICEInputsPanel
//**************************************************************************
public class ICEInputsPanel extends JPanel {

  // Static variables

  // Data
  private UintahInputPanel d_parent = null;

  // Actual Panel
  private ICEParamInputPanel iceParamInputPanel = null;

  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public ICEInputsPanel(UintahInputPanel parent) {

    // Initialize local variables
    d_parent = parent;

    // Create the panels
    iceParamInputPanel = new ICEParamInputPanel();

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create the save button
    saveButton = new JButton("Save");
    saveButton.setActionCommand("save");

    // Grid bag layout
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,0, 1,1, 5);
    gb.setConstraints(iceParamInputPanel, gbc);
    add(iceParamInputPanel);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,1, 1,1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    saveButton.addActionListener(buttonListener);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to button pressed (inner class button listener)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "save") {
        
        // Create filewriter and printwriter
        File outputFile = new File("test.ups");
        try {
          FileWriter fw = new FileWriter(outputFile);
          PrintWriter pw = new PrintWriter(fw);
     
          String tab = new String("  ");
          iceParamInputPanel.writeUintah(pw, tab);

          pw.close();
          fw.close();
        } catch (Exception event) {
          System.out.println("Could not write to file "+outputFile.getName());
        }
      }
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Time input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ICEParamInputPanel extends JPanel {

    // Data and components
    private int d_numAddHeatMat = 1;
    private String d_iceAlgo = null;
    private String d_advectAlgo = null;
    private boolean d_compatFlux = true;
    private boolean d_addHeat = true;

    private JComboBox iceAlgoComB = null;
    private JComboBox advectAlgoComB = null;
    private JCheckBox compatFluxCB = null;
    private JCheckBox addHeatCB = null;

    private DecimalField cflEntry = null;
    private WholeNumberField maxEqItEntry = null;
    private WholeNumberField minLevelEntry = null;
    private WholeNumberField maxLevelEntry = null;

    private WholeNumberField numAddHeatMatEntry = null;
    private DecimalField addHeatStartTimeEntry = null;
    private DecimalField addHeatEndTimeEntry = null;

    private JTabbedPane addHeatMatTabPane = null;
    private Vector addHeatMatls = null;
    private Vector addHeatCoeffs = null;

    public ICEParamInputPanel() {

      // Initialize
      d_iceAlgo = new String("EqForm");
      d_advectAlgo = new String("SecondOrder");
      d_compatFlux = true;
      d_addHeat = true;
      d_numAddHeatMat = 1;

      addHeatMatls = new Vector();
      addHeatCoeffs = new Vector();

      // Create a gridbaglayout and constraints
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      // Add the first panel
      JPanel panel1 = new JPanel(new GridLayout(4,0));

      JLabel iceAlgoLabel = new JLabel("Solution Technqiue");
      iceAlgoComB = new JComboBox();
      iceAlgoComB.addItem("Total Form");
      iceAlgoComB.addItem("Rate Form");
      panel1.add(iceAlgoLabel); panel1.add(iceAlgoComB); 

      JLabel advectAlgoLabel = new JLabel("Advection Algorithm");
      advectAlgoComB = new JComboBox();
      advectAlgoComB.addItem("Second Order");
      advectAlgoComB.addItem("First Order");
      panel1.add(advectAlgoLabel); panel1.add(advectAlgoComB);

      compatFluxCB = new JCheckBox("Turn on Compatible Fluxes");
      panel1.add(compatFluxCB);

      addHeatCB = new JCheckBox("Turn on Heat Addition");
      panel1.add(addHeatCB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 
                               1, 1, 5);
      gb.setConstraints(panel1, gbc);
      add(panel1); 

      // Add the second panel
      JPanel panel2 = new JPanel(new GridLayout(4,0));

      JLabel cflLabel = new JLabel("CFL Number");
      cflEntry = new DecimalField(0.25, 5);
      panel2.add(cflLabel); panel2.add(cflEntry);

      JLabel maxEqItLabel = new JLabel("Maximum Equilibrium Iterations");
      maxEqItEntry = new WholeNumberField(1000, 5);
      panel2.add(maxEqItLabel); panel2.add(maxEqItEntry);

      JLabel minLevelLabel = new JLabel("Minimum Grid Level");
      minLevelEntry = new WholeNumberField(0, 5);
      panel2.add(minLevelLabel); panel2.add(minLevelEntry);

      JLabel maxLevelLabel = new JLabel("Maximum Grid Level");
      maxLevelEntry = new WholeNumberField(1000, 5);
      panel2.add(maxLevelLabel); panel2.add(maxLevelEntry);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 1, 1, 5);
      gb.setConstraints(panel2, gbc);
      add(panel2); 

      // Add a panel (For Heat Addition)
      JPanel panel3 = new JPanel(new GridLayout(3,0));

      JLabel numAddHeatMatLabel = new JLabel("Number of Add Heat Materials");
      numAddHeatMatEntry = new WholeNumberField(1,3);
      panel3.add(numAddHeatMatLabel); panel3.add(numAddHeatMatEntry);

      JLabel addHeatStartTimeLabel = new JLabel("Add Heat Start Time");
      addHeatStartTimeEntry = new DecimalField(0.0, 9, true);
      panel3.add(addHeatStartTimeLabel); panel3.add(addHeatStartTimeEntry);

      JLabel addHeatEndTimeLabel = new JLabel("Add Heat Start Time");
      addHeatEndTimeEntry = new DecimalField(1.0e-3, 9, true);
      panel3.add(addHeatEndTimeLabel); panel3.add(addHeatEndTimeEntry);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 2, 1, 1, 5);
      gb.setConstraints(panel3, gbc);
      add(panel3); 

      // Add a tabbed pane (For Heat Addition)
      addHeatMatTabPane = new JTabbedPane();
      for (int ii = 0; ii < d_numAddHeatMat; ++ii) {

        // Create a panel
        JPanel tabPanel = new JPanel(new GridLayout(2,0));
        JLabel addHeatMatLabel = new JLabel("Material ID");
        WholeNumberField addHeatMatEntry = new WholeNumberField(1,5);
        addHeatMatls.addElement(addHeatMatEntry);
        tabPanel.add(addHeatMatLabel);
        tabPanel.add(addHeatMatEntry);
        JLabel addHeatCoeffLabel = new JLabel("Heat Coeff.");
        DecimalField addHeatCoeffEntry = new DecimalField(8.0e10, 9, true);
        addHeatCoeffs.addElement(addHeatCoeffEntry);
        tabPanel.add(addHeatCoeffLabel);
        tabPanel.add(addHeatCoeffEntry);
        String tabLabel = new String("Add Heat Mat "+ String.valueOf(ii));
        addHeatMatTabPane.addTab(tabLabel, null, tabPanel, null);
        
      }
     
      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 3, 1, 1, 5);
      gb.setConstraints(addHeatMatTabPane, gbc);
      add(addHeatMatTabPane); 

      // Create and add the listeners
      ComboBoxListener comboBoxListener = new ComboBoxListener();
      iceAlgoComB.addItemListener(comboBoxListener);
      advectAlgoComB.addItemListener(comboBoxListener);

      CheckBoxListener cbListener = new CheckBoxListener();
      compatFluxCB.addItemListener(cbListener);
      addHeatCB.addItemListener(cbListener);
     
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : ComboBoxListener
    // Purpose : Listens for item picked in combo box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class ComboBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
        // Get the combo box that has been changed
        Object source = e.getItemSelectable();

        // Get the item that has been selected
        String item = String.valueOf(e.getItem());
          
        if (source == iceAlgoComB) {
          if (item == "Total Form") {
            d_iceAlgo = "EqForm";
          } else {
            d_iceAlgo = "RateForm";
          }
        } else if (source == advectAlgoComB) {
          if (item == "First Order") {
            d_advectAlgo = "FirstOrder";
          } else {
            d_advectAlgo = "SecondOrder";
          }
        }
      }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : CheckBoxListener
    // Purpose : Listens for item seleceted in check box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class CheckBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
        // Get the combo box that has been changed
        Object source = e.getItemSelectable();

        // Get the item that has been selected
        String item = String.valueOf(e.getItem());
          
        if (source == compatFluxCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_compatFlux = true;
          } else {
            d_compatFlux = false;
          }
        } else if (source == addHeatCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_addHeat = true;
          } else {
            d_addHeat = false;
          }
        }
      }
    }

    //--------------------------------------------------------------------
    /** Write the contents out in Uintah format */
    //--------------------------------------------------------------------
    public void writeUintah(PrintWriter pw, String tab) {
      
      if (pw == null) return;

      // Write the data
      String tab1 = new String(tab+"  ");
      String tab2 = new String(tab1+"  ");
      String tab3 = new String(tab2+"  ");
      pw.println(tab+"<CFD>");
      pw.println(tab1+"<cfl> "+cflEntry.getValue()+" </cfl>");
      pw.println(tab1+"<ICE>");
      pw.println(tab2+"<max_iteration_equilibriation> "+
                      maxEqItEntry.getValue()+
                      " </max_iteration_equilibriation>");
      pw.println(tab2+"<min_grid_level> "+ minLevelEntry.getValue()+
                      " </min_grid_level> ");
      pw.println(tab2+"<max_grid_level> "+ maxLevelEntry.getValue()+
                      " </max_grid_level> ");
      pw.println(tab2+"<solution technique=\""+d_iceAlgo+"\"/>");
      pw.println(tab2+"<advection type=\""+d_advectAlgo+
                      "\" useCompatibleFluxes=\""+d_compatFlux+"\"/>");
      pw.println(tab2+"<ADD_HEAT>");
      pw.println(tab3+"<add_heat_t_start> "+addHeatStartTimeEntry.getValue()+
                      " </add_heat_t_start>");
      pw.println(tab3+"<add_heat_t_final> "+addHeatEndTimeEntry.getValue()+
                      " </add_heat_t_final>");
      pw.print(tab3+"<add_heat_matls> [");
      for (int ii = 0; ii < d_numAddHeatMat; ++ii) {      
        pw.print(((WholeNumberField) addHeatMatls.elementAt(ii)).getValue());
        if (ii < d_numAddHeatMat-1) pw.print(", ");
      }
      pw.println("] </add_heat_matls>");
      pw.print(tab3+"<add_heat_coeff> [");
      for (int ii = 0; ii < d_numAddHeatMat; ++ii) {      
        pw.print(((DecimalField) addHeatCoeffs.elementAt(ii)).getValue());
        if (ii < d_numAddHeatMat-1) pw.print(", ");
      }
      pw.println("] </add_heat_coeff>");
      pw.println(tab2+"</ADD_HEAT>");
      pw.println(tab1+"</ICE>");
      pw.println(tab+"</CFD>");
      pw.println(" ");
    }
  }

}
