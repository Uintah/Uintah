/**************************************************************************
// Class   : MPMICEExchangePanel
// Purpose : Create a panel that contains widgets to take inputs for
//           momentum and heat exchange coefficients
// Author  : Biswajit Banerjee
// Date    : 05/26/2006
// Mods    :
//**************************************************************************/

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import java.util.Vector;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableColumn;
import javax.swing.table.TableCellRenderer;
import java.text.Format;
import java.text.DecimalFormat;
import java.text.ParseException;


public class MPMICEExchangePanel extends JPanel 
                           implements ActionListener {

  // Data
  private int d_numMat = 0;
  private UintahInputPanel d_parent = null;

  private ExchangeTableModel momentumModel = null;
  private ExchangeTableModel heatModel = null;

  // Local components
  private JTable momentumTable = null;
  private JTable heatTable = null;
  private JButton updateButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public MPMICEExchangePanel(Vector mpmMat,
                       Vector iceMat,  
                       UintahInputPanel parent) {

    // Initialize local variables
    d_parent = parent;
    d_numMat = mpmMat.size()+iceMat.size();

    if (d_numMat > 6) {
      System.out.println("**ERROR**Too many materials in exchange.");
      return;
    }

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create the momentum table
    JLabel momentumLabel = new JLabel("Momentum Exchange Coefficients");
    UintahGui.setConstraints(gbc, 0, 0);
    gb.setConstraints(momentumLabel, gbc);
    add(momentumLabel);
   
    momentumModel = new ExchangeTableModel(mpmMat, iceMat, 1.0e15);
    momentumTable = new JTable(momentumModel);
    initializeTable(momentumTable);
    momentumTable.setPreferredScrollableViewportSize(new Dimension(600,100));
    momentumTable.setAutoResizeMode(JTable.AUTO_RESIZE_ALL_COLUMNS);
    momentumTable.setColumnSelectionAllowed(false);
    momentumTable.setRowSelectionAllowed(false);
    momentumTable.doLayout();
    JScrollPane momentumSP = new JScrollPane(momentumTable);
    UintahGui.setConstraints(gbc, 0, 1);
    gb.setConstraints(momentumSP, gbc);
    add(momentumSP);

    // Create the heat table
    JLabel heatLabel = new JLabel("Heat Exchange Coefficients");
    UintahGui.setConstraints(gbc, 0, 2);
    gb.setConstraints(heatLabel, gbc);
    add(heatLabel);
   
    heatModel = new ExchangeTableModel(mpmMat, iceMat, 1.0e10);
    heatTable = new JTable(heatModel);
    initializeTable(heatTable);
    heatTable.setPreferredScrollableViewportSize(new Dimension(600,100));
    heatTable.setAutoResizeMode(JTable.AUTO_RESIZE_ALL_COLUMNS);
    heatTable.setColumnSelectionAllowed(false);
    heatTable.setRowSelectionAllowed(false);
    heatTable.doLayout();
    JScrollPane heatSP = new JScrollPane(heatTable);
    UintahGui.setConstraints(gbc, 0, 3);
    gb.setConstraints(heatSP, gbc);
    add(heatSP);

    // Create the update button
    updateButton = new JButton("Update");
    updateButton.setActionCommand("update");
    UintahGui.setConstraints(gbc, 0, 4);
    gb.setConstraints(updateButton, gbc);
    add(updateButton);

    // Add listener
    updateButton.addActionListener(this);
  }

  //---------------------------------------------------------------
  // Initialize tables
  //---------------------------------------------------------------
  private void initializeTable(JTable table) {
    
    ExchangeTableModel model = (ExchangeTableModel) table.getModel();
    TableColumn col = null;
    Component comp = null;
    int headerWidth = 0;
    int cellWidth = 0;
    
    TableCellRenderer headerRenderer = 
      table.getTableHeader().getDefaultRenderer();

    // Data columns
    int maxHeaderWidth = 0;
    for (int ii=1; ii < d_numMat+1; ++ii) {
      col = table.getColumnModel().getColumn(ii);
      comp = 
        headerRenderer.getTableCellRendererComponent(null, col.getHeaderValue(),
                                                     false, false, 0, 0);
      headerWidth = comp.getPreferredSize().width;
      col.setPreferredWidth(headerWidth);
      if (headerWidth > maxHeaderWidth) maxHeaderWidth = headerWidth;
    }

    //First column
    col = table.getColumnModel().getColumn(0);
    col.setPreferredWidth(maxHeaderWidth);
  }

  //---------------------------------------------------------------
  // Update materials
  //---------------------------------------------------------------
  public void updateMaterials(Vector mpmMat, Vector iceMat) {

    d_numMat = mpmMat.size()+iceMat.size();
    if (d_numMat > 6) {
      System.out.println("**ERROR**Too many materials in exchange.");
      return;
    }
    momentumModel.updateMaterials(mpmMat, iceMat);
    heatModel.updateMaterials(mpmMat, iceMat);
    validate();
  }

  //---------------------------------------------------------------
  // Write out in Uintah format
  //---------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {

    if (pw == null) return;
    
    String tab1 = new String(tab+"  ");
    pw.println(tab+"<exchange_coefficients>");
    pw.print(tab1+"<momentum> [");
    momentumModel.writeUintah(pw, tab1);
    pw.println("] </momentum>");
    pw.print(tab1+"<heat> [");
    heatModel.writeUintah(pw, tab1);
    pw.println("] </heat>");
    pw.println(tab+"</exchange_coefficients>");

  }

  //---------------------------------------------------------------
  // Respond to button pressed 
  //---------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    if (e.getActionCommand() == "update") {


    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Table model inner classes
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ExchangeTableModel extends AbstractTableModel {

    private static final int NUMCOL = 6;
    private int d_numMat = 0;
    private String[] d_colNames = null;
    private double[] d_exchangeCoeff = null;
    private DecimalFormat formatter;

    public ExchangeTableModel(Vector mpmMat, Vector iceMat, double value) {

      d_colNames = new String[NUMCOL];
      d_exchangeCoeff = new double[NUMCOL*(NUMCOL+1)/2];

      initialize(mpmMat, iceMat, value);

      String patternExp = "0.0##E0";
      formatter = new DecimalFormat(patternExp);
    }

    public int getColumnCount() {
      return NUMCOL+1;
    }

    public int getRowCount() {
      return NUMCOL;
    }

    public String getColumnName(int col) {
      if (col == 0) {
        return new String(" ");
      } else {
        return d_colNames[col-1];
      }
    }

    public Object getValueAt(int row, int col) {
      if (col == 0) {
        return d_colNames[row];
      } else if (row > d_numMat-1 || col > d_numMat) {
        return new String(" ");
      } else {
        int actCol = col-1;
        int actRow = row;
        if (actCol < actRow) {
          int temp = actRow;
          actRow = actCol;
          actCol = temp;
        }
        actRow++;
        actCol++;
        int index = (actRow-1)*NUMCOL - actRow*(actRow-1)/2 + actCol - 1;
        //System.out.println("row = "+row+" col = "+col+" actRow = "+actRow+
        //                   " actCol = "+actCol+" index = "+index+
        //                   " value = "+d_exchangeCoeff[index]);
        return new Double(d_exchangeCoeff[index]);
      }
    }

    public void setValueAt(Object value, int row, int col) {
      if (col > 0) {
        int actCol = col-1;
        int actRow = row;
        if (actCol < actRow) {
          int temp = actRow;
          actRow = actCol;
          actCol = temp;
        }
        actRow++;
        actCol++;
        int index = (actRow-1)*NUMCOL - actRow*(actRow-1)/2 + actCol - 1;
        try {
          String input = ((String) value).toUpperCase();
          double val = formatter.parse(input).doubleValue();
          d_exchangeCoeff[index] = val;
        } catch (ParseException e) {
          System.out.println("Could not update value");
        }
        fireTableCellUpdated(row, col);
        fireTableCellUpdated(col, row);
      }
    }

    public boolean isCellEditable(int row, int col) {
      col--;
      if (col < 0 || col > d_numMat-1) return false;
      if (row >= col || row > d_numMat-1) return false;
      return true;
    }

    public void initialize(Vector mpmMat, Vector iceMat, double value) {

      d_numMat = mpmMat.size()+iceMat.size();

      for (int ii = 0; ii < NUMCOL; ++ii) {
        d_colNames[ii] = new String(" ");
      }

      int count = 0;
      for (int ii = 0; ii < mpmMat.size(); ++ii) {
        d_colNames[count++] = (String) mpmMat.elementAt(ii);
      }
      for (int ii = 0; ii < iceMat.size(); ++ii) {
        d_colNames[count++] = (String) iceMat.elementAt(ii);
      }

      count = 0;
      for (int ii = 0; ii < NUMCOL; ++ii) {
        for (int jj = ii; jj < NUMCOL; ++jj) {
          if (ii == jj) {
            d_exchangeCoeff[count++] = 0.0;
          } else {
            d_exchangeCoeff[count++] = value;
          }
        }
      }
    }

    public void updateMaterials(Vector mpmMat, Vector iceMat) {

      d_numMat = mpmMat.size()+iceMat.size();

      int count = 0;
      for (int ii = 0; ii < mpmMat.size(); ++ii) {
        d_colNames[count++] = (String) mpmMat.elementAt(ii);
      }
      for (int ii = 0; ii < iceMat.size(); ++ii) {
        d_colNames[count++] = (String) iceMat.elementAt(ii);
      }

      TableCellRenderer momTCR = 
        momentumTable.getTableHeader().getDefaultRenderer();
      TableCellRenderer heatTCR = 
        heatTable.getTableHeader().getDefaultRenderer();
      for (int col = 0; col < d_numMat; ++col) {
        TableColumn column = momentumTable.getColumnModel().getColumn(col+1);
        column.setHeaderValue(d_colNames[col]);
        Component comp = 
	  momTCR.getTableCellRendererComponent(null, d_colNames[col],
                                               false, false, 0, 0);
        column.setPreferredWidth(comp.getPreferredSize().width);
        column = heatTable.getColumnModel().getColumn(col+1);
        column.setHeaderValue(d_colNames[col]);
        column.setPreferredWidth(comp.getPreferredSize().width);
      }
    }

    public void writeUintah(PrintWriter pw, String tab) {
      if (d_exchangeCoeff.length > 0) {
        for (int ii = 0; ii < d_numMat-1; ++ii) {
          for (int jj = ii+1; jj < d_numMat; ++jj) {
            int index = ii*NUMCOL + ii*(ii-1)/2 + jj;
            pw.print(d_exchangeCoeff[index]);
            if (!(ii == d_numMat-2 && jj == d_numMat-1)) {
              pw.print(", ");
            }
          }
        }
      }
    }
  }

}
