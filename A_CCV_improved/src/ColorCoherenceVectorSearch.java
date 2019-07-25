/*
    reference from:
    [skeleton code] - http://nowokay.hatenablog.com/entry/20081008/1223496501
 */

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.*;

public class ColorCoherenceVectorSearch extends javax.swing.JFrame {

    /** Creates new form ColorCoherenceVectorSearch */
    public ColorCoherenceVectorSearch() {
        initComponents();
        for(int i = 0; i < lblResult.length; ++i){
            pnlResult.add(lblResult[i] = new JLabel());
        }
        readFile(new File("C:\\Users\\82102\\A_CCV_improved\\img_list")); // C:\search
    }

    // 해당 file의 CCV 요구
    private void readFile(File f){
        if(f.isDirectory()){
            File[] files = f.listFiles();
            for(File file : files){
                readFile(file);
            }
        }else{
            try {
                BufferedImage img = ImageIO.read(f);
                if(img == null) return;
                int[] data = colorCoherenceVector(img);
                images.put(f.getCanonicalPath(), data);
            } catch (IOException ex) {
                System.out.println(ex.getMessage());
            }

        }
    }

    JLabel[] lblResult = new JLabel[6];
    Map<String, int[]> images = new HashMap<String, int[]>();

    @SuppressWarnings("unchecked")
    private void initComponents() {

        javax.swing.JPanel jPanel1 = new javax.swing.JPanel();
        txtPath = new javax.swing.JTextField();
        javax.swing.JButton btnSearch = new javax.swing.JButton();
        pnlResult = new javax.swing.JPanel();
        lblSearch = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        txtPath.setColumns(15);
        jPanel1.add(txtPath);

        btnSearch.setText("Search");
        btnSearch.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnSearchActionPerformed(evt);
            }
        });
        jPanel1.add(btnSearch);

        getContentPane().add(jPanel1, java.awt.BorderLayout.NORTH);

        pnlResult.setLayout(new java.awt.GridLayout(2, 3));
        getContentPane().add(pnlResult, java.awt.BorderLayout.CENTER);
        getContentPane().add(lblSearch, java.awt.BorderLayout.LINE_START);

        pack();
    }

    private void btnSearchActionPerformed(java.awt.event.ActionEvent evt) {
        String filename = txtPath.getText();
        File f = new File(filename);
        try {
            BufferedImage img = ImageIO.read(f);
            showImage(lblSearch, img);
            int[] ccv = colorCoherenceVector(img);
            List<Map.Entry<String, Integer>> o = new ArrayList<Map.Entry<String, Integer>>();
            // calculate the distance
            for(Map.Entry<String, int[]> me : images.entrySet()){
                int[] comp = me.getValue();
                int dist = 0;
                for(int i = 0; i < ccv.length; ++i){
                    dist += Math.abs(ccv[i] - comp[i]);
                }
                o.add(new AbstractMap.SimpleEntry(me.getKey(), dist));
            }
            // Sort
            Collections.sort(o, new Comparator<Map.Entry<String, Integer>>() {
                @Override
                public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                    return o1.getValue() - o2.getValue();
                }
            });
            // mark
            int idx = 0;
            for(int i = 0; i < o.size(); ++i){
                if(o.get(i).getValue() == 0) continue;
                try{
                    img = ImageIO.read(new File(o.get(i).getKey()));
                    if(img == null) continue;
                }catch(Exception e){
                    continue;
                }
                showImage(lblResult[idx], img);
                idx++;
                if(idx >= 6) break;
            }
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
    }

    /** JLabel에 이미지 표시 */
    private void showImage(JLabel lbl, BufferedImage img){
        int w = img.getWidth();
        int h = img.getHeight();
        if(w < h){
            w = w * 200 / h;
            h = 200;
        }else{
            h = h * 200 / w;
            w = 200;
        }
        BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics g = im.getGraphics();
        g.drawImage(img, 0, 0, w, h, this);
        g.dispose();
        lbl.setIcon(new ImageIcon(im));
    }

    /** CCV 받아오기 */
    public static int[] colorCoherenceVector(BufferedImage imgsrc){

        int w = imgsrc.getWidth();
        int h = imgsrc.getHeight();
        // size normalization
        int limit = 200;
        if(w < h){
            w = w * limit / h;
            h = limit;
        }else{
            h = h * limit / w;
            w = limit;
        }
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D grp = (Graphics2D) img.getGraphics();
        grp.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        grp.drawImage(imgsrc, 0, 0, w, h, null);
        grp.dispose();

        // Gaussian Filter
        int[] ctemp = img.getRGB(0, 0, w, h, null, 0, w);
        int[] ctbl = new int[ctemp.length];
        int[][] filter = {
                {1, 2, 1},
                {2, 4, 2},
                {1, 2, 1}};
        for(int y = 0; y < h; ++y){
            for(int x = 0; x < w; ++x){
                int tr = 0;
                int tg = 0;
                int tb = 0;
                int t = 0;
                for(int i = -1; i < 2; ++i){
                    for(int j = -1; j < 2; ++j){
                        if(y + i < 0) continue;
                        if(x + j < 0) continue;
                        if(y + i >= h) continue;
                        if(x + j >= w) continue;
                        t += filter[i + 1][j + 1];
                        int adr = (x + j) + (y + i) * w;
                        tr += filter[i + 1][j + 1] * ((ctemp[adr] >> 16) & 255);
                        tg += filter[i + 1][j + 1] * ((ctemp[adr] >> 8)  & 255);
                        tb += filter[i + 1][j + 1] * ( ctemp[adr]        & 255);
                    }
                }
                ctbl[x + y * w] = ((tr / t) << 16) + ((tg / t) << 8) + tb / t;
            }
        }

        // color recognition
        for(int i = 0; i < ctbl.length; ++i){
            int r = (ctemp[i] >> 16) & 192;
            int g = (ctemp[i] >> 8) & 192;
            int b = ctemp[i] & 192;
            ctbl[i] = (r << 16) + (g << 8) + b;
        }

        // tag 붙임
        int[][] lbl = new int[w][h];
        int id = 0;
        for(int y = 0; y < h; ++y){
            for(int x = 0; x < w; ++x){
                int col = ctbl[y * w + x];
                if(y > 0){
                    if(x > 0){
                        if(ctbl[(y - 1) * w + x - 1] == col){
                            // with the upper left part
                            lbl[x][y] = lbl[x - 1][y - 1];
                            continue;
                        }
                    }
                    if(ctbl[(y - 1) * w + x] == col){
                        // with the upper part
                        lbl[x][y] = lbl[x][y - 1];
                        continue;
                    }
                    if(x < w - 1){
                        if(ctbl[(y - 1) * w + x + 1] == col){
                            //with the upper right part
                            lbl[x][y] = lbl[x + 1][y - 1];
                            continue;
                        }
                    }
                }
                if(x > 0){
                    if(ctbl[y * w + x - 1] == col){
                        //with the left part
                        lbl[x][y] = lbl[x - 1][y];
                        continue;
                    }
                }
                lbl[x][y] = id;
                ++id;
            }
        }

        // total add up
        int[] count = new int[id];
        int[] color = new int[id];
        for(int x = 0; x < w; ++x){
            for(int y = 0; y < h; ++y){
                count[lbl[x][y]]++;
                color[lbl[x][y]] = ctbl[y * w + x];
            }
        }

        int[] data = new int[129];
        for(int i = 0; i < id; ++i){
            int d = color[i];
            color[i] = (((d >> 22) & 3) << 4) + (((d >> 14) & 3) << 2) + ((d >> 6) & 3);
            if(count[i] < 20){
                data[color[i] * 2 + 1] ++;
            }else{
                data[color[i] * 2] ++;
            }
        }
        return data;
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new ColorCoherenceVectorSearch().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify
    private javax.swing.JLabel lblSearch;
    private javax.swing.JPanel pnlResult;
    private javax.swing.JTextField txtPath;
    // End of variables declaration

}