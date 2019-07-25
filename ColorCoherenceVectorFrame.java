/*
    reference from:

    [skeleton code]
    http://nowokay.hatenablog.com/entry/20081007/1223327913

    [paper]
    https://www.cs.cornell.edu/~rdz/Papers/PZM-MM96.pdf

 */


import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.ImageIO;
import javax.swing.*;

public class ColorCoherenceVectorFrame {
    public static void main(String[] args) throws IOException{
        JFrame f = new JFrame("CCV");
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setLayout(new GridLayout(3, 1));
        f.setSize(1200, 800); // original val - 600, 400

        JLabel l1 = new JLabel();
        JLabel l2 = new JLabel();
        JLabel l3 = new JLabel();
        f.add(l1);
        f.add(l2);
        f.add(l3);

        BufferedImage imgsrc = ImageIO.read(new File("images/amber.jpg"));
        int w = imgsrc.getWidth();
        int h = imgsrc.getHeight();

        // size normalization (이 과정에 따라 사진이 짤리게 됨..)
        // 중앙 위주로만 남고..
        int limit = 400; // original val - 200
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

        l1.setIcon(new ImageIcon(img));

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

        img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, w, h, ctbl, 0, w);
        l2.setIcon(new ImageIcon(img));

        // color recognition
        for(int i = 0; i < ctbl.length; ++i){
            int r = (ctemp[i] >> 16) & 192;
            int g = (ctemp[i] >> 8) & 192;
            int b = ctemp[i] & 192;
            ctbl[i] = (r << 16) + (g << 8) + b;
        }
        img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, w, h, ctbl, 0, w);
        l3.setIcon(new ImageIcon(img));

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

        // idx 0 ~ 63 까지의 color bin
        int[] alpha = new int[64]; // alpha = number of coherent pixels
        int[] beta = new int[64]; // beta = number of incoherent pixels
        /*
            Let us call the number of coherent pixels of the j’th discretized color αj
            and the number of incoherent pixels βj  - from the paper
         */
        for(int i = 0; i < id; ++i){
            int d = color[i];
            color[i] = (((d >> 22) & 3) << 4) + (((d >> 14) & 3) << 2) + ((d >> 6) & 3);
            if(count[i] < 20){
                beta[color[i]] ++;
            }else{
                alpha[color[i]] ++;
            }
        }

        // print out results of CCV
        for(int i = 0; i < alpha.length; ++i){
            if(alpha[i] == 0 && beta[i] == 0) continue;
            System.out.printf("%2d (%3d, %3d)%n", i, alpha[i], beta[i]);
        }

        f.setVisible(true); // display the blurred / filtered image
    }
}