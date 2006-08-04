//**************************************************************************
// Program : ICEInputsPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           ICE flags and parameters
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.util.Vector;

public class ICEInputsPanel extends JPanel {

  // Static variables

  // Data
  private Vector d_mpmMat = null;
  private Vector d_iceMat = null;

  // Actual Panel
  private ICEParamInputPanel iceParamInputPanel = null;

  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public ICEInputsPanel(Vector mpmMat, Vector iceMat,
                        UintahInputPanel parent) {

    // Initialize local variables
    d_mpmMat = mpmMat;
    d_iceMat = iceMat;

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

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
    iceParamInputPanel.refresh();
  }

  //------------------------------------------------------------------~~~~~
  // Write out in Uintah format
  //------------------------------------------------------------------~~~~~
  public void writeUintah(PrintWriter pw, String tab) {
   
    if (pw == null) return;
    iceParamInputPanel.writeUintah(pw, tab);

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
  // ICE parameters input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ICEParamInputPanel extends JPanel
                                   implements ItemListener,
                                              DocumentListener {

    // Data and components
    private int d_numAddHeatMat = 1;
    private String d_iceAlgo = null;
    private String d_advectAlgo = null;
    private boolean d_compatFlux = true;
    private boolean d_addHeat = true;
    private boolean d_clampSpVol = true;

    private JComboBox iceAlgoComB = null;
    private JComboBox advectAlgoComB = null;
    private JCheckBox compatFluxCB = null;
    private JCheckBox addHeatCB = null;
    private JCheckBox clampCB = null;

    private DecimalField cflEntry = null;
    private IntegerField maxEqItEntry = null;
    private IntegerField minLevelEntry = null;
    private IntegerField maxLevelEntry = null;

    private IntegerField numAddHeatMatEntry = null;
    private DecimalField addHeatStartTimeEntry = null;
    private DecimalField addHeatEndTimeEntry = null;

    private JTabbedPane addHeatMatTabPane = null;
    private Vector addHeatPanelList = null;

    public ICEParamInputPanel() {

      // Initialize
      d_iceAlgo = new String("EqForm");
      d_advectAlgo = new String("SecondOrder");
      d_compatFlux = true;
      d_addHeat = true;
      d_clampSpVol = true;
      d_numAddHeatMat = 1;

      addHeatPanelList = new Vector();

      // Create a gridbaglayout and constraints
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      // Add the first panel
      JPanel panel1 = new JPanel(new GridLayout(5,0));

      JLabel iceAlgoLabel = new JLabel("Solution Technqiue");
      iceAlgoComB = new JComboBox();
      iceAlgoComB.addItem("Total Form");
      iceAlgoComB.addItem("Rate Form");
      iceAlgoComB.setSelectedIndex(0);
      panel1.add(iceAlgoLabel); panel1.add(iceAlgoComB); 

      JLabel advectAlgoLabel = new JLabel("Advection Algorithm");
      advectAlgoComB = new JComboBox();
      advectAlgoComB.addItem("Second Order");
      advectAlgoComB.addItem("First Order");
      advectAlgoComB.setSelectedIndex(0);
      panel1.add(advectAlgoLabel); panel1.add(advectAlgoComB);

      compatFluxCB = new JCheckBox("Turn on Compatible Fluxes");
      compatFluxCB.setSelected(true);
      panel1.add(compatFluxCB);

      addHeatCB = new JCheckBox("Turn on Heat Addition");
      addHeatCB.setSelected(true);
      panel1.add(addHeatCB);

      clampCB = new JCheckBox("Clamp Specific Volume");
      clampCB.setSelected(true);
      panel1.add(clampCB);

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
      maxEqItEntry = new IntegerField(1000, 5);
      panel2.add(maxEqItLabel); panel2.add(maxEqItEntry);

      JLabel minLevelLabel = new JLabel("Minimum Grid Level");
      minLevelEntry = new IntegerField(0, 5);
      panel2.add(minLevelLabel); panel2.add(minLevelEntry);

      JLabel maxLevelLabel = new JLabel("Maximum Grid Level");
      maxLevelEntry = new IntegerField(1000, 5);
      panel2.add(maxLevelLabel); panel2.add(maxLevelEntry);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 1, 1, 5);
      gb.setConstraints(panel2, gbc);
      add(panel2); 

      // Add a panel (For Heat Addition)
      JPanel panel3 = new JPanel(new GridLayout(3,0));

      JLabel numAddHeatMatLabel = new JLabel("Number of Add Heat Materials");
      numAddHeatMatEntry = new IntegerField(1,3);
      numAddHeatMatEntry.getDocument().addDocumentListener(this);
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
        AddHeatPanel addHeatPanel = new AddHeatPanel();
        String tabLabel = new String("Add Heat Mat "+ String.valueOf(ii));
        addHeatMatTabPane.addTab(tabLabel, null, addHeatPanel, null);
        addHeatPanelList.addElement(addHeatPanel);
        
      }
     
      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 3, 1, 1, 5);
      gb.setConstraints(addHeatMatTabPane, gbc);
      add(addHeatMatTabPane); 

      // Create and add the listeners
      iceAlgoComB.addItemListener(this);
      advectAlgoComB.addItemListener(this);
      compatFluxCB.addItemListener(this);
      addHeatCB.addItemListener(this);
     
    }

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
      } else if (source == compatFluxCB) {
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
      } else if (source == clampCB) {
        if (e.getStateChange() == ItemEvent.SELECTED) {
          d_clampSpVol = true;
        } else {
          d_clampSpVol = false;
        }
      }
    }

    public void insertUpdate(DocumentEvent e) {
      
      int numMat = numAddHeatMatEntry.getValue();
      if (numMat <= 0) return;

      addHeatMatTabPane.removeAll();
      for (int ii=0; ii < numMat; ++ii){
        AddHeatPanel addHeatPanel = new AddHeatPanel();
        String tabLabel = new String("Add Heat Mat "+ String.valueOf(ii));
        addHeatMatTabPane.addTab(tabLabel, null, addHeatPanel, null);
        addHeatPanelList.addElement(addHeatPanel);
      }
      d_numAddHeatMat = numMat;
      validate();
    }

    public void removeUpdate(DocumentEvent e) {
    }

    public void changedUpdate(DocumentEvent e) {
    }

    public void refresh() {
      int numMat = numAddHeatMatEntry.getValue();
      for (int ii=0; ii < numMat; ++ii){
        ((AddHeatPanel) addHeatPanelList.elementAt(ii)).updateMatList();
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
      pw.println(tab2+"<max_iteration_equilibration> "+
                 maxEqItEntry.getValue()+
                 " </max_iteration_equilibration>");
      pw.println(tab2+"<min_grid_level> "+ minLevelEntry.getValue()+
                 " </min_grid_level> ");
      pw.println(tab2+"<max_grid_level> "+ maxLevelEntry.getValue()+
                 " </max_grid_level> ");
      pw.println(tab2+"<solution technique=\""+d_iceAlgo+"\"/>");
      pw.println(tab2+"<advection type=\""+d_advectAlgo+
                 "\" useCompatibleFluxes=\""+d_compatFlux+"\"/>");
      if (d_clampSpVol) {
        pw.println(tab2+"<ClampSpecificVolume> "+d_clampSpVol+
                   " </ClampSpecificVolume> ");
      }
      if (d_addHeat) {
        pw.println(tab2+"<ADD_HEAT>");
        pw.println(tab3+"<add_heat_t_start> "+addHeatStartTimeEntry.getValue()+
                   " </add_heat_t_start>");
        pw.println(tab3+"<add_heat_t_final> "+addHeatEndTimeEntry.getValue()+
                   " </add_heat_t_final>");
        pw.print(tab3+"<add_heat_matls> [");
        for (int ii = 0; ii < d_numAddHeatMat; ++ii) {      
          pw.print(((AddHeatPanel) addHeatPanelList.elementAt(ii)).getMatID());
          if (ii < d_numAddHeatMat-1) pw.print(", ");
        }
        pw.println("] </add_heat_matls>");
        pw.print(tab3+"<add_heat_coeff> [");
        for (int ii = 0; ii < d_numAddHeatMat; ++ii) {      
          pw.print(((AddHeatPanel) addHeatPanelList.elementAt(ii)).getCoeff());
          if (ii < d_numAddHeatMat-1) pw.print(", ");
        }
        pw.println("] </add_heat_coeff>");
        pw.println(tab2+"</ADD_HEAT>");
      }
      pw.println(tab1+"</ICE>");
      pw.println(tab+"</CFD>");
      pw.println(" ");
    }
  }

  private class AddHeatPanel extends JPanel
                             implements ListSelectionListener {

    private int d_numMPM;
    private int d_numICE;
    private int d_addHeatMatID;
    private double d_addHeatCoeff;

    private DefaultListModel addHeatMatListModel = null;
    private JList addHeatMatList = null;
    private JScrollPane addHeatMatSP = null;
    private DecimalField addHeatCoeffEntry = null;

    public AddHeatPanel() {

      d_numMPM = d_mpmMat.size();
      d_numICE = d_iceMat.size();

      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      JLabel addHeatMatLabel = new JLabel("Material ID");
      addHeatMatListModel = new DefaultListModel();
      for (int ii = 0; ii < d_numICE; ++ii) {
        addHeatMatListModel.addElement(d_iceMat.elementAt(ii));
      }
      addHeatMatList = new JList(addHeatMatListModel);
      addHeatMatList.setSelectedIndex(0);
      addHeatMatList.setVisibleRowCount(2);
      addHeatMatList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
      addHeatMatList.addListSelectionListener(this);
      addHeatMatSP = new JScrollPane(addHeatMatList);

      UintahGui.setConstraints(gbc, 0, 0);
      gb.setConstraints(addHeatMatLabel, gbc);
      add(addHeatMatLabel);
      UintahGui.setConstraints(gbc, 1, 0);
      gb.setConstraints(addHeatMatSP, gbc);
      add(addHeatMatSP);

      d_addHeatMatID = d_numMPM;

      JLabel addHeatCoeffLabel = new JLabel("Heat Coeff.");
      addHeatCoeffEntry = new DecimalField(8.0e10, 9, true);

      UintahGui.setConstraints(gbc, 0, 1);
      gb.setConstraints(addHeatCoeffLabel, gbc);
      add(addHeatCoeffLabel);
      UintahGui.setConstraints(gbc, 1, 1);
      gb.setConstraints(addHeatCoeffEntry, gbc);
      add(addHeatCoeffEntry);

      d_addHeatCoeff = 8.0e10;
    }

    public void valueChanged(ListSelectionEvent e) {
      d_addHeatMatID = d_numMPM + addHeatMatList.getSelectedIndex();
    }

    public int getMatID() {
      return d_addHeatMatID;
    }

    public double getCoeff() {
      return d_addHeatCoeff;
    }

    public void updateMatList() {
      if (d_iceMat.size() != d_numICE || d_mpmMat.size() != d_numMPM) {
        d_numMPM = d_mpmMat.size();
        d_numICE = d_iceMat.size();
        addHeatMatListModel.removeAllElements();
        for (int ii = 0; ii < d_numICE; ++ii) {
          addHeatMatListModel.addElement(d_iceMat.elementAt(ii));
        }
        validate();
      }
    }
  } // Add heat panel

}
